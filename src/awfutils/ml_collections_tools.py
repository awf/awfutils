from ml_collections import config_flags, config_dict


def update_from_argv(config: config_dict.ConfigDict, prefix: str = "config") -> None:
    """
    Given a ConfigDict with fields such as:
        option1: 4.5
        option2: "fred"

    Examine sys.argv for entries like
      --prefix.option1=5.4
      --prefix.option2="conor"%%!

    And update (i.e. overwrite) entries in the dict.

    Typical usage is that you fill the config with defaults, and then
    call `update_from_argv` to overwrite the defaults with command-line arguments

    ```py
      config = ml_collections.config_dict.ConfigDict()
      config.size = 4.5
      config.name = "fred"

      # ... read from command line
      update_from_argv(config)

      #... now use the values...
      print(config.name, " has size ", config.size)
    ```

    The default `prefix` is "config", this can be overridden.
    """
    config_flags.DEFINE_config_dict(prefix, config)
    import sys
    import absl

    absl.flags.FLAGS(sys.argv)
