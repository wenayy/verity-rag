# Postmortem: RouteIQ Solver Outage, 14 March 2025

Severity: SEV1. Duration: 2 hours 47 minutes, from 06:12 to 08:59 IST on 14 March 2025.
Impact: RouteIQ could not generate new routes for 31 customers during the morning dispatch window. Existing routes kept working in the driver app.
Root cause: a schema migration added a non-null column to the routes table in the sharded CockroachDB cluster without a default value. The solver's write path failed and the retry storm exhausted the connection pool.
Detection: the on-call engineer, Vikram Sethi, was paged by a Grafana alert on solver error rate at 06:19, seven minutes after the first failures.
Resolution: the migration was rolled back and the connection pool was restarted region by region. Ana Costa acted as incident commander.
Customer impact: Kaveri Foods and Deccan Agro were the two most affected Enterprise and Growth customers; Kaveri Foods received service credits of 10% of its March fee under its SLA.
Action items: (1) all migrations must pass an online-schema-change linter in CI, (2) the solver now uses exponential back-off with jitter, (3) a fallback to the Google OR-Tools solver was added for degraded mode.
