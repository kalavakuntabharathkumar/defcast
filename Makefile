.PHONY: demo test forecast

demo:
	python -m pipeline.make_demo_data
test:
	pytest -q
forecast:
	python -m pipeline.forecast
