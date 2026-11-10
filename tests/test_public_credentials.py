import importlib.util
import unittest

class PublicCredentialTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("context_ledger.live.keys"), "Public environment-only provider has not been supplied")
        from context_ledger.live import keys
        self.keys = keys

    def test_missing_input_fails_closed(self):
        with self.assertRaises(self.keys.KeyUnavailable):
            self.keys.EnvironmentKeyProvider(environ={}).get()

    def test_explicit_environment_value_is_wrapped_and_not_printed(self):
        value = "unit-test-placeholder"
        provider = self.keys.EnvironmentKeyProvider(environ={"CONTEXT_LEDGER_RESEARCH_API_KEY": value})
        secret = provider.get()
        self.assertEqual(secret.reveal(), value)
        self.assertNotIn(value, repr(secret))
        self.assertEqual(provider.profile, "research")

    def test_profiles_do_not_share_credential_values(self):
        provider = self.keys.EnvironmentKeyProvider("aperture", environ={"CONTEXT_LEDGER_RESEARCH_API_KEY": "unit-test-placeholder"})
        with self.assertRaises(self.keys.KeyUnavailable):
            provider.get()

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            self.keys.EnvironmentKeyProvider("not-a-profile", environ={})
