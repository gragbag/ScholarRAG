-- Create the auxiliary database Langfuse needs, alongside the app database
-- (which POSTGRES_DB already creates). Runs once on first container init.
SELECT 'CREATE DATABASE langfuse'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec
