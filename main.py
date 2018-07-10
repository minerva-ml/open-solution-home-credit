import click

from src.pipeline_manager import PipelineManager

pipeline_manager = PipelineManager()


@click.group()
def main():
    pass


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train(pipeline_name, dev_mode):
    pipeline_manager.train(pipeline_name, dev_mode)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate(pipeline_name, dev_mode):
    pipeline_manager.evaluate(pipeline_name, dev_mode)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
def predict(pipeline_name, dev_mode, submit_predictions):
    pipeline_manager.predict(pipeline_name, dev_mode, submit_predictions)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate_predict(pipeline_name, submit_predictions, dev_mode):
    pipeline_manager.train(pipeline_name, dev_mode)
    pipeline_manager.evaluate(pipeline_name, dev_mode)
    pipeline_manager.predict(pipeline_name, dev_mode, submit_predictions)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate(pipeline_name, dev_mode):
    pipeline_manager.train(pipeline_name, dev_mode)
    pipeline_manager.evaluate(pipeline_name, dev_mode)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate_predict(pipeline_name, submit_predictions, dev_mode):
    pipeline_manager.evaluate(pipeline_name, dev_mode)
    pipeline_manager.predict(pipeline_name, dev_mode, submit_predictions)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-l', '--model_level', help='level of modeling first or second are available', default='first',
              required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate_cv(pipeline_name, model_level, dev_mode):
    pipeline_manager.train_evaluate_cv(pipeline_name, model_level, dev_mode)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-l', '--model_level', help='level of modeling first or second are available', default='first',
              required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate_predict_cv(pipeline_name, model_level, dev_mode, submit_predictions):
    pipeline_manager.train_evaluate_predict_cv(pipeline_name, model_level, dev_mode, submit_predictions)


if __name__ == "__main__":
    main()
