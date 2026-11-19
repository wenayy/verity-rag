# RouteIQ Technical Architecture

RouteIQ is split into three services: the Planner API (Python, FastAPI), the Solver (Rust, a custom adaptive large neighbourhood search) and the Tracker (Go), which streams driver GPS positions.
Orders enter through the Planner API, are batched every 90 seconds and sent to the Solver. Solutions are cached in Redis and persisted to a sharded CockroachDB cluster.
The ETA model used by RouteIQ and Pulse is a gradient-boosted model trained by Rohan Das on 48 million historical stops; it reduced mean absolute ETA error by 31% compared with the previous heuristic.
Dynamic re-routing listens for traffic events from HERE Maps and order changes; a re-plan is triggered when the expected delay on a route exceeds 12 minutes.
All services run on Kubernetes (Amazon EKS) in the ap-south-1 (Mumbai) and eu-central-1 (Frankfurt) regions. Infrastructure is managed with Terraform.
