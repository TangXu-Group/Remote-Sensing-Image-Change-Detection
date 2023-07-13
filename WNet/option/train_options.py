from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--num_epochs', type=int, default=125)
        self.parser.add_argument('--num_decay_epochs', type=int, default=125)
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--initialize', type=bool, default=True, help='')

        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs.')
        self.parser.add_argument('--save_latest_freq', type=int, default=10000,
                            help='frequency of saving the latest results.')
        self.parser.add_argument('--display_freq', type=int, default=2000,
                            help='frequency of showing training results on screen.')
        self.parser.add_argument('--print_freq', type=int, default=2000,
                            help='frequency of showing training results on console.')
        self.isTrain = True
