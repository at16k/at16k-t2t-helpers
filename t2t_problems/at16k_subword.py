"""
at16k (Subword)
"""

from tensor2tensor.utils import registry
from . import asr


@registry.register_problem()
class At16kSubword(asr.AsrProblem):
    """
    at16k (Subword)
    """

    @property
    def is_8k(self):
        """Is the audio recorded at 8k?"""
        return False

    @property
    def multiprocess_generate(self):
        """Whether to generate the data in multiple parallel processes."""
        return True

    @property
    def num_generate_tasks(self):
        """Needed if multiprocess_generate is True."""
        return self.num_train_shards + self.num_dev_shards + self.num_test_shards

    @property
    def approx_vocab_size(self):
        return 1000

    @property
    def num_train_shards(self):
        """
        Number of training shards
        """
        return 15000

    @property
    def num_dev_shards(self):
        """
        Number of dev shards
        """
        return 250

    @property
    def num_test_shards(self):
        """
        Number of test shards
        """
        return 250

    @property
    def use_train_shards_for_dev(self):
        """If true, we only generate training data and hold out shards for dev."""
        return False

    def split_data_equally(self, a, n):
        k, m = divmod(len(a), n)
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    def prepare_to_generate(self, data_dir, tmp_dir):
        return

    def generator(self, data_dir, tmp_dir, task_id=-1):
        assert 0 <= task_id < self.num_generate_tasks
        return

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        assert 0 <= task_id < self.num_generate_tasks
        return
