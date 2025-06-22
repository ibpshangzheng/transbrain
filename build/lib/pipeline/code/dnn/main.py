# main.py
from pipeline.code.dnn.config.parse import Config
from pipeline.code.dnn.config.parse import parse_arguments
from pipeline.code.dnn.config.logger import init_logger, get_logger
from pipeline import DnnPipeline

def execute_pipeline():

    args = parse_arguments()
    config = Config()

    config.update_parameters(
        train = args.train,
        trans = args.trans,
        independent_test = args.independent_test,
        repeat_n = args.repeat_n,
        input_units = args.input_units,
        hidden_units1 = args.hidden_units1,
        hidden_units2 = args.hidden_units2,
        hidden_units3 = args.hidden_units3,
        output_units = args.output_units,
        weight_decay = args.weight_decay,
        max_epochs = args.max_epochs,
        learning_rate = args.learning_rate,
        batch_size = args.batch_size,
        valid_ratio = args.valid_ratio,
        use_bestrank = args.use_bestrank,
        use_bestvalid = args.use_bestvalid,
        shuffle = args.shuffle
    )

    config.update_paths(
        gene_path = args.gene_path,
        data_path = args.data_path,
        save_path = args.save_path,
        mouse_trans_path = args.mouse_trans_path,
        human_trans_path = args.human_trans_path,
        independent_data_path = args.independent_data_path,
        independent_s_path = args.independent_s_path,
        independent_test_path = args.independent_test_path,
        human_mouse_path = args.human_mouse_path
    )

    if config.independent_test:
        init_logger(log_dir=config.data_files['independent_s_path'],log_file='train_dnn_independent.log')
    else:
        init_logger(log_dir=config.data_files['save_path'],log_file='train_dnn.log')
        
    logger = get_logger()

    logger.info("====== Starting Pipeline ======")

    try:
        
        pipeline = DnnPipeline(config=config)
        pipeline.execute()
        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
            logger.error(e)
            raise 

if __name__ == '__main__':

    execute_pipeline()
