.PHONY: init download embed training-data

# install Poetry and Python dependencies
init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install


# backfills the feature group using historical data
training-data:
	poetry run python scripts/generate_training_data.py