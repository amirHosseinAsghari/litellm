use axum::{
    Json,
    http::{HeaderName, HeaderValue, StatusCode},
    response::{AppendHeaders, IntoResponse, Response},
};
use serde_json::{Value, json};

const MESSAGES: &str = "/v1/messages";

/// Python's `ProxyRateLimitError`: a limit on the caller refused the call before it ran.
#[derive(Debug, PartialEq)]
pub struct RateLimited {
    pub message: String,
    /// In the order Python sends them.
    pub headers: Vec<(String, String)>,
}

impl RateLimited {
    /// The 429 the route matched at `path` answers: the Anthropic envelope on `/v1/messages`, the
    /// OpenAI one everywhere else.
    pub fn response(self, path: &str) -> Response {
        let headers: Vec<(HeaderName, HeaderValue)> = self
            .headers
            .into_iter()
            .filter_map(|(name, value)| {
                Some((
                    HeaderName::try_from(name).ok()?,
                    HeaderValue::try_from(value).ok()?,
                ))
            })
            .collect();
        (
            StatusCode::TOO_MANY_REQUESTS,
            AppendHeaders(headers),
            Json(body(path, self.message)),
        )
            .into_response()
    }
}

fn body(path: &str, message: String) -> Value {
    if path == MESSAGES {
        return json!({"type": "error", "error": {
            "type": "rate_limit_error",
            "message": message,
            "litellm_error_type": "throttling_error",
        }});
    }
    json!({"error": {
        "message": message,
        "type": "throttling_error",
        "param": null,
        "code": "429",
    }})
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::messages(
        "/v1/messages",
        json!({"type": "error", "error": {
            "type": "rate_limit_error",
            "message": "over",
            "litellm_error_type": "throttling_error",
        }})
    )]
    #[case::chat_completions(
        "/v1/chat/completions",
        json!({"error": {"message": "over", "type": "throttling_error", "param": null, "code": "429"}})
    )]
    #[tokio::test]
    async fn a_refusal_is_a_429_in_the_routes_own_envelope_with_its_headers(
        #[case] path: &str,
        #[case] expected: Value,
    ) {
        let response = RateLimited {
            message: "over".into(),
            headers: vec![
                ("retry-after".into(), "60".into()),
                ("rate_limited_by".into(), "api_key".into()),
            ],
        }
        .response(path);

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["retry-after"], "60");
        assert_eq!(response.headers()["rate_limited_by"], "api_key");
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), expected);
    }
}
