"""User authentication — Google OAuth verification + our own session JWTs.

The ``pyjwt`` / ``google-auth`` imports live INSIDE functions (the ``auth``
extra is absent in core-only CI), so importing this package is always safe.
"""
