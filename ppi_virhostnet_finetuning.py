# %% Imports
from argparse import ArgumentParser
from torch.utils.data.dataloader import DataLoader
from src.modeling.ProtBertPPIModel import ProtBertPPIModel
from typing import List
import pandas as pd
from pandas.core.frame import DataFrame
from transformers import BertTokenizer,T5Tokenizer
from pytorch_lightning import Trainer, seed_everything


from src import settings
from src import npe_ppi_logger
from src.data.VirHostNetDataset import VirHostNetData

logger = npe_ppi_logger.get_custom_logger(name=__name__)

def generate_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--train_csv",
        default=settings.BASE_DATA_DIR + "/generated/sarscov2/ml/train.txt",
        # default=settings.BASE_DATA_DIR + "/generated/vp1/ml/train.txt",
        type=str,
        help="Path to the file containing the train data.",
    )
    parser.add_argument(
        "--predict_csv",
        default=settings.BASE_DATA_DIR + "/generated/sarscov2/ml/test.txt",
        # default=settings.BASE_DATA_DIR + "/generated/vp1/ml/predict_vp1_interactions_template.txt",
        type=str,
        help="Path to the file containing the inferencing data.",
    )
    parser.add_argument(
        "--perform_training", default=True, type=bool, help="Perform training."
    )
    parser.add_argument(
        "--prediction_checkpoint",
        default=settings.BASE_MODELS_DIR + "/sarscov2_ppi_model2.ckpt",
        type=str,
        help="File path of checkpoint to be used for prediction."
    )

    return parser

def prepare_params():
    
    logger = npe_ppi_logger.get_custom_logger(name=__name__)

    logger.info("Starting parsing arguments...")
    parser = generate_parser()
    params = parser.parse_args()
    logger.info("Finishing parsing arguments.")

    return params

# %% Predict
def main(params):

    if params.perform_training == True:
        logger.info("Starting training.")
        
        model_name = "prot_t5_xl_bfd"
        tokenizer_name = "prot_t5_xl_bfd"
        save_ckpt = settings.BASE_MODELS_DIR + "/prot_t5_xl_bfd.ckpt"
        target_col = 'class'
        seq_col_a = 'sequenceA'
        seq_col_b = 'sequenceB'
        max_len = 1536
        batch_size = 8
        seed = 1
        seed_everything(seed)
        
        model_params = {}
        # TODO: update params from paper
        model_params["encoder_learning_rate"] = 5e-06
        model_params["warmup_steps"] = 200
        model_params["max_epochs"] = 20
        model_params["min_epochs"] = 5
        model_params["local_logger"] = logger

        model_params["adam_epsilon"] = 5.90539e-08
        model_params["per_device_train_batch_size"] = 8
        model_params["per_device_eval_batch_size"] = 8
        model_params["per_device_test_batch_size"] = 8 
        model_params["per_device_predict_batch_size"] = 8
        model_params["max_length"] = 1536
        model_params["encoder_learning_rate"] = 5e-06
        # model_params["learning_rate"] = 0.000429331
        model_params["learning_rate"] = 0.01
        model_params["weight_decay"] = 6.34672e-07
        model_params["dropout_prob"] = 0.5
        model_params["nr_frozen_epochs"] = 0
        model_params["gradient_checkpointing"] = True
        model_params["label_set"] = "1,0"
        # model_params["weight_decay"] = 1e-2
        model_params["train_csv"] = settings.BASE_DATA_DIR + "/generated/sarscov2/ml/train.txt"
        model_params["valid_csv"] = settings.BASE_DATA_DIR + "/generated/sarscov2/ml/valid.txt"
        model_params["test_csv"] = settings.BASE_DATA_DIR + "/generated/sarscov2/ml/test.txt"
        model_params["loader_workers"] = 8
        model_params["model_name"] = model_name
        model_params["tokenizer_name"] = tokenizer_name

        model = ProtBertPPIModel(model_params)
        trainer = Trainer(
            accelerator='gpu', devices=-1,
            max_epochs=model_params["max_epochs"],
            # callbacks=callbacks, 
            # checkpoint_callback=checkpoint_callback,
            # progress_bar_refresh_rate=5,
            num_sanity_val_steps=0,
            #logger = npe_ppi_logger.get_mlflow_logger_for_PL(trial.study.study_name)
        )

        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name, do_lower_case=False, local_files_only=True)
        #  Read dataset
        # data: DataFrame = pd.read_csv(settings.BASE_DATA_DIR + "/generated/sarscov2/ml/all_data.txt", sep = "\t", header=0)
        # train_set, val_set, test_set = torch.utils.data.random_split(data, [536, 66, 66])
        # train_set.dataset.iloc[train_set.indices].to_csv(settings.BASE_DATA_DIR + "/generated/sarscov2/ml/train.txt", header=True, index=None,
        #                          sep='\t')
        # val_set.dataset.iloc[val_set.indices].to_csv(settings.BASE_DATA_DIR + "/generated/sarscov2/ml/valid.txt", header=True, index=None, sep='\t')
        #
        # test_set.dataset.iloc[test_set.indices].to_csv(settings.BASE_DATA_DIR + "/generated/sarscov2/ml/test.txt", header=True, index=None,
        #                          sep='\t')

        train_data: DataFrame = pd.read_csv(model_params["train_csv"], sep="\t", header=0)
        dataset = VirHostNetData(train_data,
            tokenizer = tokenizer, 
            max_len = max_len, 
            seq_col_a = seq_col_a, 
            seq_col_b = seq_col_b, 
            target_col = target_col
        )
        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8) # type:ignore

        data: DataFrame = pd.read_csv(model_params["valid_csv"], sep = "\t", header=0)
        dataset = VirHostNetData(data,
            tokenizer = tokenizer,
            max_len = max_len,
            seq_col_a = seq_col_a,
            seq_col_b = seq_col_b,
            target_col = target_col
        )
        valid_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8) # type:ignore

        # data: DataFrame = pd.read_csv(model_params["test_csv"], sep = "\t", header=0)
        # dataset = VirHostNetData(data,
        #     tokenizer = tokenizer,
        #     max_len = max_len,
        #     seq_col_a = seq_col_a,
        #     seq_col_b = seq_col_b,
        #     target_col = target_col
        # )
        # test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8) # type:ignore

        trainer.fit(model, train_loader, valid_loader)
        trainer.save_checkpoint(save_ckpt)
        
        logger.info("Finishing training.")

    else:

        df_to_output = pd.read_csv(params.predict_csv, sep="\t", header=0)

        # Load model
        logger.info("Loading model.")

        model: ProtBertPPIModel = ProtBertPPIModel.load_from_checkpoint(
            params.prediction_checkpoint, 
        )

        # Predict
        logger.info("Loading dataset.")
        dataset = VirHostNetData(
            df=df_to_output, 
            tokenizer=model.tokenizer, 
            max_len=1536, 
            # seq_col_a="receptor_protein_sequence",
            seq_col_a="sequenceA",
            # seq_col_b="spike_protein_sequence",
            seq_col_b="sequenceB",
            target_col=False
        )
            
        logger.info("Predicting.")
        trainer = Trainer(accelerator='gpu', devices=-1, deterministic=True)
        predict_dataloader = DataLoader(dataset, num_workers=8) #type: ignore
        predictions = trainer.predict(model=model, dataloaders=predict_dataloader, return_predictions=True)
        for ix in range(len(predictions)): # type:ignore
            score = predictions[ix]['probability'][0] # type:ignore
            logger.info("For entry %s, we found a score of %s", ix, score)
            df_to_output.at[ix, "score"] = score
            
        # Save results
        results_file = settings.BASE_DATA_DIR + "/generated/sarscov2/ml/test_predict.txt"
        df_to_output.to_csv(results_file, sep="\t", index=False)

#%%
if __name__ == '__main__':
    params = prepare_params()
    main(params)
