//! Which requests the gateway answers itself in front of the Python proxy.
//!
//! [`Admit`] holds what the native app was built from, and each request is put to the three stages
//! of checks in `checks`. This is the only part that does IO: it authenticates the caller before
//! the `CALLER` stage and reads the body with `body` before the `BODY` stage.

mod body;
mod checks;
mod groups;

use std::sync::Arc;

use axum::{body::Body, extract::Request, http::request::Parts};
use litellm_config::{Config, Environment};
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_env::Lookup;
use litellm_gateway_auth::{Auth, BearerOrXApiKey};
use litellm_gateway_inference::JsonObject;
use litellm_router_core::registry::DeploymentFilter;
use serde_json::{Map, Value};

use crate::AppState;
use checks::Reason;
use groups::Groups;

/// Why the gateway does not serve a request the way the Python proxy would.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error, strum::IntoStaticStr)]
pub enum Refused {
    #[error("the gateway is not running the proxy's configuration")]
    #[strum(serialize = "configuration")]
    Configuration,
    #[error("the proxy's configuration has a setting the gateway does not honor yet")]
    #[strum(serialize = "settings")]
    Settings,
    #[error("the gateway does not serve this route")]
    #[strum(serialize = "route")]
    Route,
    #[error("the gateway cannot check this caller's credential")]
    #[strum(serialize = "credential")]
    Credential,
    #[error("the request carries a header only the proxy reads")]
    #[strum(serialize = "headers")]
    Headers,
    #[error("the request body carries something only the proxy reads")]
    #[strum(serialize = "body")]
    Body,
    #[error("the gateway does not serve this model the way the proxy does")]
    #[strum(serialize = "model")]
    Model,
}

/// One deployment the router holds: its model group and its provider, or the provider's name when
/// `LlmProviders` does not list it.
type Deployed = (String, Result<LlmProviders, String>);

/// Admits the requests the gateway serves the way the Python proxy would, and refuses every other
/// one, giving it back as the client sent it, with why.
///
/// It is only right about the native app built from the same config, so a new config needs both
/// built again and replaced together.
pub struct Admit {
    config: Config,
    /// The variables only the environment holds, as `main` read them.
    environment: Environment,
    /// What gives the proxy a database, which it reads keys, spend and settings from.
    database: Option<Box<str>>,
    /// Each deployment the router built from `config`, since only the router knows which provider
    /// serves each one.
    deployed: Vec<Deployed>,
    /// `None` when a fallback list or an alias is written in a shape that is not read.
    groups: Option<Groups>,
    auth: Auth,
}

/// The variables the proxy builds a database from when `database_url` names none.
const DATABASE_VARIABLES: [&str; 4] = [
    "DATABASE_URL",
    "DATABASE_HOST",
    "IAM_TOKEN_DB_AUTH",
    "AZURE_POSTGRESQL_AUTH",
];

fn database(config: &Config, environment: &dyn Lookup) -> Option<Box<str>> {
    config
        .general_settings
        .database_url
        .as_ref()
        .map(|_| "general_settings.database_url".into())
        .or_else(|| {
            DATABASE_VARIABLES
                .into_iter()
                .find(|variable| environment.get(variable).is_some())
                .map(Box::from)
        })
}

fn refused(request: Parts, body: Body, reason: Reason) -> (Box<Request>, Refused) {
    tracing::debug!(why = %reason.why, over = %reason.over, "python answers a request over this");
    (Box::new(Request::from_parts(request, body)), reason.why)
}

impl Admit {
    /// By the `config` the native app `state` is built from, with its environment under it as
    /// `load_config` leaves it, and the `environment` `main` read.
    pub fn new(config: &Config, environment: &Environment, state: &AppState) -> Self {
        let deployed = state
            .inference
            .router
            .registry()
            .deployments(&DeploymentFilter::default())
            .iter()
            .map(|deployment| {
                let provider = deployment.provider();
                (
                    deployment.model_name.clone(),
                    provider
                        .parse::<LlmProviders>()
                        .map_err(|_| provider.to_owned()),
                )
            })
            .collect();
        Self {
            environment: environment.clone(),
            database: database(config, &*state.environment),
            ..Self::over(config, deployed)
        }
    }

    fn over(config: &Config, deployed: Vec<Deployed>) -> Self {
        Self {
            groups: Groups::read(
                config
                    .model_list
                    .iter()
                    .map(|model| model.model_name.as_str()),
                &config.router_settings,
            ),
            auth: Auth::from_config(config).with_extractor(Arc::new(BearerOrXApiKey)),
            config: config.clone(),
            environment: Environment::default(),
            database: None,
            deployed,
        }
    }

    /// The body is read only for a request that passes every other check, and parsed once: an
    /// admitted request carries it parsed, as the [`JsonObject`] the route takes. A refused request
    /// keeps the bytes the client sent and carries nothing more.
    pub async fn admit(
        self: Arc<Self>,
        request: Request,
    ) -> Result<Request, (Box<Request>, Refused)> {
        let (mut parts, body) = request.into_parts();
        if let Some(reason) = checks::HEAD.iter().find_map(|check| check(&self, &parts)) {
            return Err(refused(parts, body, reason));
        }
        let authenticated = self.auth.authenticate_parts(&parts).await;
        if let Some(reason) = checks::CALLER
            .iter()
            .find_map(|check| check(&self, &authenticated))
        {
            return Err(refused(parts, body, reason));
        }
        let (body, whole) = body::read(&parts, body).await;
        let Some(fields) =
            whole.and_then(|sent| serde_json::from_slice::<Map<String, Value>>(&sent).ok())
        else {
            return Err(refused(
                parts,
                body,
                Reason::new(Refused::Body, "unreadable"),
            ));
        };
        if let Some(reason) = checks::BODY.iter().find_map(|check| check(&self, &fields)) {
            return Err(refused(parts, body, reason));
        }
        parts.extensions.insert(JsonObject(fields));
        Ok(Request::from_parts(parts, body))
    }
}

#[cfg(test)]
mod tests {
    use axum::http::Method;
    use rstest::rstest;

    use super::*;

    const CONFIG: &str = "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\n  - model_name: gpt\n    litellm_params:\n      model: openai/gpt-5.5\n      api_key: provider-key\ngeneral_settings:\n  master_key: gateway-key\n";
    const MASTER_KEY: (&str, &str) = ("authorization", "Bearer gateway-key");

    fn admit() -> Arc<Admit> {
        let config = Config::from_yaml(CONFIG).unwrap();
        let deployed = vec![
            ("claude".to_owned(), Ok(LlmProviders::Anthropic)),
            ("gpt".to_owned(), Ok(LlmProviders::Openai)),
        ];
        Arc::new(Admit::over(&config, deployed))
    }

    fn messages(headers: &[(&str, &str)], model: &str) -> Request {
        headers
            .iter()
            .fold(Request::post("/v1/messages"), |request, (name, value)| {
                request.header(*name, *value)
            })
            .body(Body::from(format!(
                r#"{{"model":"{model}","max_tokens":8,"messages":[]}}"#
            )))
            .unwrap()
    }

    #[rstest]
    #[case::over_its_caller("Bearer sk-someone-else", "claude", Refused::Credential)]
    #[case::over_its_body("Bearer gateway-key", "claude\",\"user\":\"end-user", Refused::Body)]
    #[case::over_a_body_that_is_not_json("Bearer gateway-key", "claude\"}}", Refused::Body)]
    #[case::over_its_model("Bearer gateway-key", "gpt", Refused::Model)]
    #[tokio::test]
    async fn a_refused_request_comes_back_as_the_client_sent_it(
        #[case] authorization: &str,
        #[case] model: &str,
        #[case] why: Refused,
    ) {
        let sent = format!(r#"{{"model":"{model}","max_tokens":8,"messages":[]}}"#);
        let request = Request::post("/v1/messages?beta=true")
            .header("authorization", authorization)
            .header("x-litellm-call-id", "abc")
            .body(Body::from(sent.clone()))
            .unwrap();

        let (returned, refused) = admit().admit(request).await.unwrap_err();

        assert_eq!(
            (
                refused,
                returned.method(),
                returned.uri().to_string().as_str(),
                returned.headers().len(),
                returned.headers()["x-litellm-call-id"].to_str().unwrap(),
                returned.extensions().len(),
            ),
            (why, &Method::POST, "/v1/messages?beta=true", 2, "abc", 0)
        );
        let body = axum::body::to_bytes(returned.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body, sent);
    }

    #[rstest]
    #[tokio::test]
    async fn an_admitted_request_keeps_its_body_and_carries_it_parsed() {
        let sent = r#"{"model":"claude","max_tokens":8,"messages":[]}"#;

        let admitted = admit()
            .admit(messages(&[MASTER_KEY], "claude"))
            .await
            .unwrap();

        let parsed = admitted
            .extensions()
            .get::<JsonObject>()
            .map(|body| &body.0);
        assert_eq!(
            parsed,
            serde_json::from_str::<Map<String, Value>>(sent)
                .ok()
                .as_ref()
        );
        let body = axum::body::to_bytes(admitted.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body, sent);
    }
}
