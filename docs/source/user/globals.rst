Globals and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

``import gpubackendtools`` initializes the :class:`~gpubackendtools.globals.Globals`
singleton: it reads the configuration, sets up the ``"gpubackendtools"`` logger (records up to
``INFO`` go to stdout, higher levels to stderr) and creates the backend registry, which the
package then fills with its ``gbt_*`` backends. The backend lookup functions are documented on
:doc:`gpubackendtools`.

Options are read, in increasing priority, from their defaults, an INI file with a
``[gpubackendtools]`` section, ``GPUBACKENDTOOLS_*`` environment variables, CLI-style
arguments and setter values. The import passes no CLI arguments, so environment variables
and the INI file are the ways to configure a normal session:

.. list-table::
   :header-rows: 1

   * - Option
     - Environment variable
     - INI key
     - Default
   * - ``log_level``: ``logging`` level name, any case (``info``); ``set_log_level`` also
       takes an int
     - ``GPUBACKENDTOOLS_LOG_LEVEL``
     - ``log-level``
     - ``WARNING``
   * - ``log_format``: :class:`logging.Formatter` format string
     - ``GPUBACKENDTOOLS_LOG_FORMAT``
     - ``log-format``
     - ``None`` (the ``logging`` default)
   * - ``config_file``: INI file to read (must exist)
     - ``GPUBACKENDTOOLS_CONFIG_FILE``
     - (none)
     - the first of ``./gpubackendtools.ini``, the user config directory and the site config
       directory (:func:`~gpubackendtools.utils.config.detect_cfg_file`)
   * - ``ignore_cfg``: skip the INI file
     - ``GPUBACKENDTOOLS_IGNORE_CFG_FILE``
     - (none)
     - ``False``

.. warning::

   ``enabled_backends`` (``GPUBACKENDTOOLS_ENABLED_BACKENDS``, INI key ``enabled-backends``)
   has no effect on the registry, and a non-empty value fails validation with ``NameError``
   (at import, for the environment variable and the INI key; even an empty environment
   variable counts as non-empty). An unknown ``log_level`` name currently fails with
   ``AttributeError`` instead of a configuration error.

To re-initialize with other values, reset the globals through a setter and re-publish the
registered backend classes, since the reset leaves the registry empty:

.. code-block:: python

    from gpubackendtools.globals import Globals, get_config_setter

    get_config_setter(reset=True).set_log_level("info").finalize()
    Globals().backends_manager.add_backends({})

Objects built before a reset hold backend names that resolve again only after that
``add_backends`` call.

.. automodule:: gpubackendtools.globals

.. autoclass:: gpubackendtools.globals.Globals
    :members:

.. autoclass:: gpubackendtools.globals.ConfigurationSetter
    :members:
    :exclude-members: set_storage_path, set_file_registry_path, set_file_download_path,
        enable_file_download, disable_file_download, set_file_integrity_check,
        disable_file_tags, add_file_extra_paths

.. autofunction:: gpubackendtools.globals.get_logger

.. autofunction:: gpubackendtools.globals.get_config

.. autofunction:: gpubackendtools.globals.get_config_setter

.. autofunction:: gpubackendtools.globals.reset

.. autofunction:: gpubackendtools.globals.initialize

.. autoclass:: gpubackendtools.globals.GPUBACKENDTOOLSGlobalsInitializedTwice
    :show-inheritance:

.. autoclass:: gpubackendtools.globals.GPUBACKENDTOOLSGlobalsReadOnly
    :show-inheritance:

Configuration machinery
-----------------------

.. automodule:: gpubackendtools.utils.config

.. autoclass:: gpubackendtools.utils.config.Configuration
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.utils.config.InitialConfigConsumer
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.utils.config.ConfigConsumer
    :members:

.. autoclass:: gpubackendtools.utils.config.ConfigEntry
    :members:

.. autoclass:: gpubackendtools.utils.config.ConfigItem

.. autoclass:: gpubackendtools.utils.config.ConfigSource
    :members:

.. autofunction:: gpubackendtools.utils.config.detect_cfg_file

.. autofunction:: gpubackendtools.utils.config.userstr_to_bool

.. autofunction:: gpubackendtools.utils.config.userinput_to_pathlist

.. autofunction:: gpubackendtools.utils.config.userinput_to_strlist

.. autofunction:: gpubackendtools.utils.config.compatibility_isinstance
