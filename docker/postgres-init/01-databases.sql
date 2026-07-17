-- Create the auxiliary databases alongside the app database (which POSTGRES_DB
-- already creates). Runs once on first container init.
--   langfuse         — the Langfuse tracing service's own DB
--   scholarrag_test  — the test database (kept separate so `make test` never
--                      wipes the dev data in `scholarrag`)
SELECT 'CREATE DATABASE langfuse'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec

SELECT 'CREATE DATABASE scholarrag_test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'scholarrag_test')\gexec
