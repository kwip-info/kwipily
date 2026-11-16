# Build and verification tools

Start with `setup.sh` and `build-app.sh` for the development app.
`component_bundles.py --check` verifies generated components and their source assets.
`generate_data02_learning.py --check` verifies generated teaching resources.

Runtime, broker and signing helpers document how the official app is assembled.
Publisher release operators require private signing credentials and acceptance receipts;
they do not grant access to KWIP infrastructure and are not needed for local builds.
