from pathlib import Path
from typing import Literal
from lightning.pytorch import LightningDataModule
from xlstm_mixer.data_provider.data_factory import data_provider, data_dict
from .enums import Task, TimeFreq, ForecastingTaskOptions
from torch.utils.data import DataLoader, Dataset
from xlstm_mixer.data_provider.uea import collate_fn


class TSLibDataModule(LightningDataModule):
    def __init__(
        self,
        task: Task,
        dataset_name: str,
        batch_size: int = 128,
        num_workers: int = 4,
                persistent_workers: bool = False,
        root_path: Path = Path("/common-ts/datasets/tslib_datasets"),
        augmentation_ratio: int = 0,
        test_mode: bool = False,
    ):
        super().__init__()

        self.task = task
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.root_path = root_path
        self.augmentation_ratio = augmentation_ratio
        self.enc_in = 0
        self.test_mode = test_mode

        self.configure_dataset_specifics()

        # self.save_hyperparameters()

    @property
    def task_name(self):
        return self.task.value

    @property
    def data(self) -> str:
        # TODO handle m4 and classification etc
        if self.dataset_name in data_dict:
            return self.dataset_name
        if "PEMS" in self.dataset_name:
            return "PEMS"
        elif self.task == Task.CLASSIFICATION:
            return "UEA"
        else:
            return "custom"

    def setup(self, stage: str) -> None:

        train_flag = "train"
        val_flag = "val"  # "test" #"val"
        if "PEMS" in self.dataset_name or self.test_mode:
            val_flag = "test"
        test_flag = "test"

        if self.task == Task.CLASSIFICATION:
            train_flag = "TRAIN"
            val_flag = "TEST"
            test_flag = "TEST"

        match stage:
            case "fit":
                self.train_dataset, _ = data_provider(self, train_flag)
                self.val_dataset, _ = data_provider(self, val_flag)
            case "validate":
                self.val_dataset, _ = data_provider(self, val_flag)
            case "test":
                self.test_dataset, _ = data_provider(self, test_flag)
            case _:
                raise NotImplementedError(f"Stage {stage} not supported")

    def _make_dataloader(self, dataset: Dataset, shuffle: bool = True):
        if self.task == Task.CLASSIFICATION:

            assert hasattr(
                self, "seq_len"
            ), "seq_len must be defined for classification task, this is a bug"
            col_fn = lambda x: collate_fn(x, max_len=self.seq_len)
        else:
            col_fn = None

        if not self.task.is_shuffable:
            shuffle = False

        return DataLoader(
            dataset,
            self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=col_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def configure_dataset_specifics(self):
        self.data_path: str | None = None

        if self.task == Task.CLASSIFICATION:
            self.root_path = str(self.root_path / self.dataset_name)
            # this string cast is needed for Shenanigans in tslib
            return

        match self.dataset_name:
            case "PEMS03":
                self.enc_in = 358
                self.root_path = self.root_path / "PEMS"
                self.data_path = "PEMS03.npz"
            case "PEMS04":
                self.enc_in = 307
                self.root_path = self.root_path / "PEMS"
                self.data_path = "PEMS04.npz"
            case "PEMS07":
                self.enc_in = 883
                self.root_path = self.root_path / "PEMS"
                self.data_path = "PEMS07.npz"
            case "PEMS08":
                self.enc_in = 170
                self.root_path = self.root_path / "PEMS"
                self.data_path = "PEMS08.npz"
            case "Weather":
                self.enc_in = 21
                self.root_path = self.root_path / "weather"
                self.data_path = "weather.csv"
            case "Traffic":
                self.enc_in = 862
                self.root_path /= "traffic"
                self.data_path = "traffic.csv"
            case "Electricity":
                self.enc_in = 321
                self.root_path /= "electricity"
                self.data_path = "electricity.csv"
            case "ETTh1":
                self.enc_in = 7
                self.root_path /= "ETT-small"
                self.data_path = "ETTh1.csv"
            case "ETTh2":
                self.enc_in = 7
                self.root_path /= "ETT-small"
                self.data_path = "ETTh2.csv"
            case "ETTm1":
                self.enc_in = 7
                self.root_path /= "ETT-small"
                self.data_path = "ETTm1.csv"
            case "ETTm2":
                self.enc_in = 7
                self.root_path /= "ETT-small"
                self.data_path = "ETTm2.csv"
            case "Ili":
                self.enc_in = 9  # TODO check
                self.root_path /= "illness"
                self.data_path = "national_illness.csv"
            case "Solar":
                self.enc_in = 137
                self.root_path /= "solar"
                self.data_path = "solar_AL.txt"
            case "Exchange":
                self.enc_in = 137
                self.root_path /= "exchange_rate"
                self.data_path = "exchange_rate.csv"
            case _:
                raise UserWarning(f"Dataset {self.dataset_name} not supported")

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataset, shuffle=False)


class ForecastingTSLibDataModule(TSLibDataModule):
    def __init__(
        self,
        dataset_name: Literal[
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "Weather",
            "Electricity",
            "Traffic",
            "Ili",
            "Solar",
            "PEMS03",
            "PEMS04",
            "PEMS07",
            "PEMS08",
        ],
        task: Task = Task.LONG_TERM_FORECAST,
        batch_size: int = 128,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        num_workers: int = 0,#4,
        persistent_workers: bool = False,
        embed: Literal["timeF", "fixed", "learned"] = "timeF",
        freq: TimeFreq = TimeFreq.HOURLY,
        root_path: Path = Path("/common-ts/datasets/tslib_datasets"),
        seasonal_patterns: str = "Monthly",
        forecasting_option: ForecastingTaskOptions = ForecastingTaskOptions.MULTIVARIATE_2_MULTIVARIATE,
        target: str = "OT",
        augmentation_ratio: int = 0,
        test_mode: bool = False,
    ):
        super().__init__(
            
            task, dataset_name, batch_size, num_workers,persistent_workers, root_path, augmentation_ratio, test_mode=test_mode
        )
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.embed = embed
        self.freq_kind = freq
        self.seasonal_patterns = seasonal_patterns
        self.forecasting_option = forecasting_option
        self.target = target

        # self.save_hyperparameters()

    @property
    def freq(self):
        return self.freq_kind.value

    @property
    def features(self) -> Literal["M", "MS", "S"]:
        return self.forecasting_option.value


class ClassificationTSLibDataModule(TSLibDataModule):
    def __init__(
        self,
        dataset_name: Literal[
            "EthanolConcentration",
            "FaceDetection",
            "Handwriting",
            "Heartbeat",
            "JapaneseVowels",
            "PEMS-SF",
            "SelfRegulationSCP1",
            "SelfRegulationSCP2",
            "SpokenArabicDigits",
            "UWaveGestureLibrary",
        ],
        task: Task = Task.CLASSIFICATION,
        seq_len: int = 96,
        num_workers: int = 0,
        root_path: Path = Path("/common-ts/datasets/tslib_datasets"),
        augmentation_ratio: int = 0,
    ):
        super().__init__(
            task,
            dataset_name,
            num_workers=num_workers,
            root_path=root_path,
            augmentation_ratio=augmentation_ratio,
        )
        self.seq_len = seq_len
        self.embed = "NOTUSED"
        self.freq = "NOTUSED"

        # self.save_hyperparameters()
