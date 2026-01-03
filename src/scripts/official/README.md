# Official public training scripts

Please check the config carefully before attempting to run them. You may need to adjust hyperparameters based on your hardware.

## Usage

Each Python training script in this directory has the same CLI, and they're intended to be launched directly with `torchrun` or, for Beaker users, through OLMo-core Beaker launch CLI: `python -m olmo_core.launch.beaker`.
The scripts themselves take several required arguments as well as any number of config overrides in dot-notation.
Run a script with the `--help` flag to see which arguments are required, and run with the `--dry-run` flag to see the full config that will be used.
To override a field in the config such as the `data_loader`'s `prefetch_factor`, you could add the option `--data_loader.prefetch_factor=4` to your command-line options.


#   官方公开训练脚本

在尝试运行这些脚本之前，请仔细检查配置文件。你可能需要根据你的硬件调整超参数。

#   使用方法

本目录下的每个 Python 训练脚本都具有相同的命令行接口（CLI），可以直接通过 torchrun 启动，或者对于 Beaker 用户，通过 OLMo-core 的 Beaker 启动 CLI 启动：python -m olmo_core.launch.beaker。
这些脚本本身需要若干必填参数，同时也支持任意数量的点式（dot-notation）配置覆盖。
使用 --help 参数运行脚本，可以查看哪些参数是必填的；使用 --dry-run 参数运行，可以查看将要使用的完整配置。
例如，如果你想覆盖配置中的某个字段，比如 data_loader 的 prefetch_factor，可以在命令行中添加选项：--data_loader.prefetch_factor=4。