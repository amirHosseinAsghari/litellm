use super::{Admit, Authenticated, Reason, Refused};

/// The gateway checks the master key and no other credential, so a virtual key, a JWT or no key
/// at all goes to Python.
pub(super) fn caller(_: &Admit, authenticated: &Authenticated) -> Option<Reason> {
    authenticated
        .is_err()
        .then(|| Reason::new(Refused::Credential, "unverified"))
}
