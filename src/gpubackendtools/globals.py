"""Definition of global states (logger, config, fast backend, ...).

:class:`Globals` is the process-wide singleton holding the ``gpubackendtools`` logger, the
:class:`~gpubackendtools.utils.config.Configuration` and the
:class:`~gpubackendtools.gpubackendtools.BackendsManager` registry shared by every package
of the stack. It is created when this module is imported and fully initialized on the first
access to :attr:`Globals.backends_manager`, which ``import gpubackendtools`` performs when it
registers the ``gbt_*`` backends.

Module-level entry points:

- :func:`get_backend`, :func:`has_backend`, :func:`get_first_backend`: backend lookup by
  registry name (``"<pkg>_<flavor>"``), with the ``"<pkg>_cuda"`` / ``"<pkg>_gpu"`` aliases
  in :func:`get_backend`.
- :func:`get_logger`, :func:`get_config`: the shared logger and configuration.
- :func:`get_config_setter`, :func:`initialize`, :func:`reset`: configuration lifecycle.

Example:
    >>> import gpubackendtools
    >>> from gpubackendtools.globals import get_backend, has_backend
    >>> if has_backend("gbt_cpu"):
    ...     backend = get_backend("gbt_cpu")
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import typing

from .utils.config import (
    ConfigConsumer,
    ConfigEntry,
    ConfigSource,
    Configuration,
    InitialConfigConsumer,
    detect_cfg_file,
)
from .exceptions import GPUBACKENDTOOLSException

from .gpubackendtools import BackendsManager


class Singleton(type):
    """Metaclass making each class a singleton: ``Cls()`` always returns the first instance.

    Constructor arguments of later calls are ignored.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GPUBACKENDTOOLSGlobalsInitializedTwice(GPUBACKENDTOOLSException):
    """Exception raised if globals are initialized multiple times."""


class GPUBACKENDTOOLSGlobalsReadOnly(GPUBACKENDTOOLSException):
    """Exception raised when trying to modify the global structure."""


class MultiHandlerTarget:
    """Helper class to transition logger from memory-buffer to stream handler during globals init.

    Used as the flush target of the pre-init ``MemoryHandler``: forwards each buffered
    record at or above ``level`` to every handler.

    Args:
        level: Minimum record level forwarded.
        *handlers: Handlers receiving the records.
    """

    handlers: typing.List[logging.Handler]
    level: int

    def __init__(self, level: int, *handlers):
        self.handlers = handlers
        self.level = level

    def handle(self, record):
        """Pass ``record`` to every handler if its level is at least :attr:`level`."""
        if record.levelno >= self.level:
            for handler in self.handlers:
                handler.handle(record)


class ConfigurationSetter:
    """Helper class to define configuration options.

    Collects option values (each converted with its
    :class:`~gpubackendtools.utils.config.ConfigEntry` converter) and hands them to
    ``finalizer`` on :meth:`finalize`; the globals' setter re-initializes
    :class:`Globals` with them as the highest-priority source. Every setter method returns
    ``self`` so calls chain.

    Only the labels present in
    :meth:`~gpubackendtools.utils.config.Configuration.config_entries` work:
    ``enabled_backends``, ``log_level`` and ``log_format``. The ``file_*`` setters raise
    ``KeyError`` because those entries are commented out.

    Args:
        finalizer: Called with :meth:`get_args` by :meth:`finalize`. Default ``None``
            (``finalize`` does nothing).

    Warning:
        :class:`Globals` is already initialized once ``gpubackendtools`` is imported, so a
        setter is only available through ``get_config_setter(reset=True)``, and the reset
        leaves the backend registry empty until the next
        :meth:`~gpubackendtools.gpubackendtools.BackendsManager.add_backends` call.

    Example:
        .. code-block:: python

            from gpubackendtools.globals import Globals, get_config_setter

            get_config_setter(reset=True).set_log_level("info").finalize()
            # Re-publish the already known backend classes into the new registry.
            Globals().backends_manager.add_backends({})
    """

    _args: typing.Dict[str, typing.Any]
    _entries: typing.Dict[str, ConfigEntry]
    _finalizer: typing.Optional[typing.Callable[[typing.Dict[str, typing.Any]], None]]

    def __init__(
        self,
        finalizer: typing.Optional[
            typing.Callable[[typing.Dict[str, typing.Any]], None]
        ] = None,
    ):
        self._args = {}
        self._entries = {entry.label: entry for entry in Configuration.config_entries()}
        self._finalizer = finalizer

    def enable_backends(self, *backends: str) -> ConfigurationSetter:
        """Enable one or multiple backends.

        Args:
            *backends: Backend names (lower-cased by the converter).

        Returns:
            ConfigurationSetter: ``self``.

        Warning:
            The value is not effective: validating it at :meth:`finalize` references the
            undefined name ``KNOWN_BACKENDS`` (``NameError``), and
            :class:`~gpubackendtools.gpubackendtools.BackendsManager` ignores
            ``enabled_backends``.
        """
        return self._convert_and_set("enabled_backends", backends)

    def set_log_level(self, level: typing.Union[str, int]) -> ConfigurationSetter:
        """Set a specific log level.

        Args:
            level: ``logging`` level as an int or a name (``"debug"``, ``"INFO"``, ...).

        Returns:
            ConfigurationSetter: ``self``.

        Raises:
            AttributeError: For an unknown level name: the intended
                ``exceptions.ConfigurationValidationError`` is not defined in
                :mod:`gpubackendtools.exceptions`.
        """
        return self._convert_and_set("log_level", level)

    # TODO check: keep/remove/document -- KeyError: file_* config entry commented out; no callers
    def set_storage_path(self, path: os.PathLike) -> ConfigurationSetter:
        """Modify the storage path"""
        return self._convert_and_set("file_storage_path", path)

    def set_log_format(self, format: str) -> ConfigurationSetter:
        """Change the log format.

        Args:
            format: :class:`logging.Formatter` format string for the stdout / stderr handlers.

        Returns:
            ConfigurationSetter: ``self``.
        """
        return self._convert_and_set("log_format", format)

    # TODO check: keep/remove/document -- KeyError: file_* config entry commented out; no callers
    def set_file_registry_path(self, registry_path: os.PathLike) -> ConfigurationSetter:
        """Set the file registry to use"""
        return self._convert_and_set("file_registry_path", registry_path)

    # TODO check: keep/remove/document -- KeyError: file_* config entry commented out; no callers
    def set_file_download_path(self, path: os.PathLike) -> ConfigurationSetter:
        """Set the download path"""
        return self._convert_and_set("file_download_path", path)

    # TODO check: keep/remove/document -- KeyError: file_* config entry commented out; no callers
    def enable_file_download(self) -> ConfigurationSetter:
        """Authorize the file manager to download missing files"""
        return self._convert_and_set("file_allow_download", True)

    # TODO check: keep/remove/document -- KeyError: file_* config entry commented out; no callers
    def disable_file_download(self) -> ConfigurationSetter:
        """Forbid the file manager to download missing files"""
        return self._convert_and_set("file_allow_download", False)

    # TODO check: keep/remove/document -- KeyError: file_* config entry commented out; no callers
    def set_file_integrity_check(self, when: str) -> ConfigurationSetter:
        """Define when integrity checks should be performed (never, once, always)"""
        return self._convert_and_set("file_integrity_check", when)

    # TODO check: keep/remove/document -- KeyError: file_* config entry commented out; no callers
    def disable_file_tags(self, *tags: str) -> ConfigurationSetter:
        """Disable files associated to given tags"""
        return self._convert_and_set("file_disabled_tags", tags)

    # TODO check: keep/remove/document -- KeyError: file_* config entry commented out; no callers
    def add_file_extra_paths(
        self, *paths: typing.List[os.PathLike]
    ) -> ConfigurationSetter:
        """Add supplementary research paths to file manager"""
        return self._convert_and_set("file_extra_paths", paths)

    def _convert_and_set(self, label: str, value: typing.Any) -> ConfigurationSetter:
        """Convert ``value`` with the entry's converter and store it under ``label``.

        Logs a warning if :class:`Globals` is already initialized.

        Raises:
            KeyError: If ``label`` is not a :class:`Configuration` entry.
        """
        if Globals().is_initialized:
            get_logger().warning(
                "GPUBACKENDTOOLS configurations is already initialized. Option {}={} might not be taken into account".format(
                    label, value
                )
            )
        self._args[label] = self._entries[label].convert(value)
        return self

    def get_args(self) -> dict[str, typing.Any]:
        """Get a dictionnary of set options.

        Returns:
            dict[str, Any]: Converted values keyed by configuration label.
        """
        return self._args

    def finalize(self):
        """Finalize GPUBACKENDTOOLS initialization with specified parameters.

        Calls the finalizer with :meth:`get_args`; for the globals' setter this is
        ``Globals().init(set_args=...)``, which raises
        :class:`GPUBACKENDTOOLSGlobalsInitializedTwice` if the globals are initialized.
        """
        if self._finalizer is not None:
            self._finalizer(self.get_args())


class Globals(metaclass=Singleton):
    """Process-wide singleton holding the logger, the configuration and the backend registry.

    ``Globals()`` always returns the same instance (created when
    :mod:`gpubackendtools.globals` is imported). Initialization is two-phase: construction
    pre-initializes the logger (records buffered in memory) and a
    :class:`ConfigurationSetter`; :meth:`init` then reads the configuration (setter values,
    CLI arguments, ``GPUBACKENDTOOLS_*`` environment variables, ``gpubackendtools.ini``),
    attaches stdout / stderr log handlers and creates the
    :class:`~gpubackendtools.gpubackendtools.BackendsManager`. :meth:`init` runs implicitly
    on the first access to :attr:`config` or :attr:`backends_manager`; ``import
    gpubackendtools`` triggers it.

    Attributes cannot be set from outside (:class:`GPUBACKENDTOOLSGlobalsReadOnly`); every
    package reaches the shared registry through :attr:`backends_manager`.

    Example:
        .. code-block:: python

            from gpubackendtools import Globals

            Globals().backends_manager.add_backends({"mypkg_cpu": MyPkgCpuBackend})
            Globals().logger.info("registered")
    """

    _logger: logging.Logger
    _initial_config: InitialConfigConsumer
    _config: Configuration
    # _file_manager: FileManager
    _config_setter: ConfigurationSetter
    _backends_manager: BackendsManager

    _to_initialize: bool

    def __init__(self):
        """Pre-initialize the logger and the configuration setter."""
        self._preinit_logger()
        self._preinit_setter()
        super().__setattr__("_to_initialize", True)

    def init(
        self,
        cli_args: typing.Optional[typing.Sequence[typing.Any]] = None,
        set_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ):
        """Initialize config, logger and backend manager with optional CLI arguments.

        After this call the configuration setter is dropped and :attr:`is_initialized` is
        ``True``. The new :class:`~gpubackendtools.gpubackendtools.BackendsManager` has an
        empty registry until the next ``add_backends`` call.

        Args:
            cli_args: Command-line style arguments (``["--log-level", "debug", "-v"]``).
                Default ``None`` (no CLI options).
            set_args: Option values keyed by configuration label, highest priority.
                Default ``None`` uses the values collected by the current setter.

        Raises:
            GPUBACKENDTOOLSGlobalsInitializedTwice: If already initialized.
        """
        if not super().__getattribute__("_to_initialize"):
            raise GPUBACKENDTOOLSGlobalsInitializedTwice(
                "GPUBACKENDTOOLS globals are already initialized."
            )
        if set_args is None:
            config_setter = self.get_configuration_setter()
            set_args = config_setter.get_args()
        self._init_config(cli_args=cli_args, set_args=set_args)
        self._postconfig_logger()
        # self._init_file_manager()
        self._init_backends_manager()

        super().__setattr__("_to_initialize", False)
        super().__setattr__("_config_setter", None)

        self.logger.debug("GPUBACKENDTOOLS globals initialized.")

    def reset(self):
        """Reset the global structure.

        Drops the configuration, logger and setter and returns to the pre-initialized
        state. The next access to :attr:`backends_manager` re-initializes and creates a
        new manager whose registry stays empty until the next ``add_backends`` call
        (registered classes are kept in the class-level ``_known_backends``, so
        ``Globals().backends_manager.add_backends({})`` re-publishes them).
        """

        # Remove attributes existing only in initialized globals
        if self.is_initialized:
            super().__delattr__("_initial_config")
            super().__delattr__("_config")
            # super().__delattr__("_file_manager")

        # Remove attributes always existing
        super().__delattr__("_logger")
        super().__delattr__("_config_setter")
        super().__delattr__("_to_initialize")

        # Reinitiliaze the structure
        Globals.__init__(self)

    @property
    def is_initialized(self) -> bool:
        """Whether global properties are initialized."""
        return not super().__getattribute__("_to_initialize")

    @property
    def logger(self) -> logging.Logger:
        """The ``"gpubackendtools"`` :class:`logging.Logger`."""
        return super().__getattribute__("_logger")

    @property
    def config(self) -> Configuration:
        """The :class:`~gpubackendtools.utils.config.Configuration` (initializes if needed)."""
        if not self.is_initialized:
            self.init()
        return super().__getattribute__("_config")

    # @property
    # def file_manager(self) -> FileManager:
    #     if not self.is_initialized:
    #         self.init()
    #     return super().__getattribute__("_file_manager")

    @property
    def backends_manager(self) -> BackendsManager:
        """The shared backend registry (initializes the globals on first access).

        Returns:
            BackendsManager: The :class:`~gpubackendtools.gpubackendtools.BackendsManager`.
        """
        if super().__getattribute__("_to_initialize"):
            self.init()
        return super().__getattribute__("_backends_manager")

    def get_configuration_setter(self, reset: bool = False) -> ConfigurationSetter:
        """
        Access a configuration setter.

        Args:
            reset: If ``True`` and the globals are initialized, :meth:`reset` them first.
                Default ``False``.

        Returns:
            ConfigurationSetter: The setter whose :meth:`~ConfigurationSetter.finalize`
            initializes the globals.

        Raises:
            GPUBACKENDTOOLSGlobalsInitializedTwice: If globals are already initialized and
                ``reset`` is ``False`` (always the case after ``import gpubackendtools``).
        """
        if self.is_initialized:
            if reset:
                self.reset()
            else:
                raise GPUBACKENDTOOLSGlobalsInitializedTwice(
                    "GPUBACKENDTOOLS globals are already initialized. Cannot access a setter."
                )
        return super().__getattribute__("_config_setter")

    def __setattr__(self, name, value):
        raise GPUBACKENDTOOLSGlobalsReadOnly(
            "Cannot set attribute on Globals structure."
        )

    def _preinit_logger(self):
        """Pre-initialize logger: buffer up to 1024 records in a ``MemoryHandler``."""
        logger = logging.getLogger("gpubackendtools")
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            logger.removeHandler(handler)

        INITIAL_CAPACITY = 1024  # Log up to 1024 messages until globals are initialized
        handler = logging.handlers.MemoryHandler(capacity=INITIAL_CAPACITY)
        handler.set_name("_gpubackendtools_initial_handler")
        logger.addHandler(handler)
        super().__setattr__("_logger", logger)

    def _preinit_setter(self):
        """Initialize the configuration setter."""
        setter = ConfigurationSetter(finalizer=lambda args: self.init(set_args=args))
        super().__setattr__("_config_setter", setter)

    def _init_config(self, cli_args, set_args):
        """Initialize configurations.

        Two :class:`~gpubackendtools.utils.config.InitialConfigConsumer` passes read the
        ``--ignore-env`` / ``--ignore-config-file`` / ``--config-file`` options: the first
        from the CLI only (to learn ``--ignore-env``), the second from the CLI plus
        ``os.environ`` unless ignored. The config file is then the explicit one, else
        :func:`~gpubackendtools.utils.config.detect_cfg_file` (none when ``ignore_cfg`` is
        set, by CLI or env), and the full
        :class:`~gpubackendtools.utils.config.Configuration` is built from that file, the
        env vars and CLI arguments the second pass did not consume, and ``set_args``.
        """
        import os

        ignores_cfg = InitialConfigConsumer(
            cli_args=cli_args
        )  # Read only CLI args (ignores)

        logger = get_logger()
        if ignores_cfg.ignore_env:
            logger.debug(
                "ConfigInitialization: ignoring environment as requested by command-line options"
            )

        file_cfg = InitialConfigConsumer(
            env_vars=None if ignores_cfg.ignore_env else os.environ, cli_args=cli_args
        )  # Read CLI args (and env if not ignored)

        if file_cfg.ignore_cfg:
            logger.debug(
                "ConfigInitialization: ignoring configuration file as requested by {} options".format(
                    "command-line"
                    if file_cfg.get_item("ignore_cfg")[0].source == ConfigSource.CLIOPT
                    else "environment"
                )
            )

        cfg_file = (
            None
            if file_cfg.ignore_cfg
            else file_cfg.config_file
            if file_cfg.config_file is not None
            else detect_cfg_file()
        )

        if cfg_file is not None:
            logger.debug(
                "ConfigInitialization: using configuration file '{}'.".format(cfg_file)
            )

        _, extra_env_vars, extra_cli_args = file_cfg.get_extras()

        config = Configuration(
            config_file=cfg_file,
            env_vars=extra_env_vars,
            cli_args=extra_cli_args,
            set_args=set_args,
        )
        super().__setattr__("_initial_config", file_cfg)
        super().__setattr__("_config", config)

        logger.debug("ConfigInitialization: final configuration entries are")
        self._log_config(file_cfg)
        self._log_config(config)

    def _log_config(self, config: ConfigConsumer, log_level=logging.DEBUG):
        """Print configuration options in logs"""
        logger = get_logger()
        for item, entry in config.get_items():
            logger.log(
                level=log_level,
                msg=f" {entry.label}={item.value} (from: {str(item.source)})",
            )

    def _postconfig_logger(self):
        """Initialize logger after config is initialized.

        Records ``<= INFO`` go to stdout and ``> INFO`` to stderr, at the configured
        ``log_level`` / ``log_format``; buffered pre-init records are flushed through them.
        """
        import sys

        logger: logging.Logger = super().__getattribute__("_logger")
        cfg: Configuration = super().__getattribute__("_config")

        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

        stderr_handler = logging.StreamHandler(stream=sys.stderr)
        stderr_handler.addFilter(lambda record: record.levelno > logging.INFO)

        if cfg.log_format is not None:
            formatter = logging.Formatter(fmt=cfg.log_format)
            stdout_handler.setFormatter(formatter)
            stderr_handler.setFormatter(formatter)

        for handler in logger.handlers:
            if handler.get_name() == "_gpubackendtools_initial_handler":
                assert isinstance(handler, logging.handlers.MemoryHandler)
                handler.setTarget(
                    MultiHandlerTarget(cfg.log_level, stdout_handler, stderr_handler)
                )
                handler.close()
                break

        logger.handlers.clear()
        logger.setLevel(cfg.log_level)
        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)

    # def _init_file_manager(self):
    #     cfg: Configuration = super().__getattribute__("_config")
    #     file_manager = FileManager(cfg)
    #     super().__setattr__("_file_manager", file_manager)

    def _init_backends_manager(self):
        """Create a new, empty :class:`~gpubackendtools.gpubackendtools.BackendsManager`."""
        cfg: Configuration = super().__getattribute__("_config")
        backends_manager = BackendsManager(enabled_backends=cfg.enabled_backends)
        super().__setattr__("_backends_manager", backends_manager)


def get_logger() -> logging.Logger:
    """Get GPUBACKENDTOOLS logger.

    Returns:
        logging.Logger: The ``"gpubackendtools"`` logger held by :class:`Globals`.
    """
    return Globals().logger


# def get_file_manager() -> FileManager:
#     """Get GPUBACKENDTOOLS File Manager"""
#     return Globals().file_manager


def get_config() -> Configuration:
    """Get GPUBACKENDTOOLS configuration.

    Returns:
        Configuration: The :class:`~gpubackendtools.utils.config.Configuration` held by
        :class:`Globals` (initializing the globals if needed).
    """
    return Globals().config


# TODO check: keep/remove/document -- AttributeError (Globals has no add_backends); no callers
def add_backends(available_backends: typing.Dict[str, Backend]):
    """
    Add a backend to the singleton instance.

    Currently raises ``AttributeError``: :class:`Globals` has no ``add_backends`` method.
    Use ``Globals().backends_manager.add_backends(...)``.
    """
    Globals().add_backends(available_backends=available_backends)


def get_backend(backend_name: str) -> Backend:
    """
    Get a backend by its name.

    ``backend_name`` is a full registry name, ``"<pkg>_<flavor>"`` (``"gbt_cpu"``,
    ``"lisatools_cuda12x"``, ``"gbt_jax"``). Two aliases are resolved first, on the
    ``_``-separated tokens of the name:

    - a ``"cuda"`` token (``"<pkg>_cuda"``) returns the first available of
      ``"<pkg>_cuda13x"``, ``"<pkg>_cuda12x"``, ``"<pkg>_cuda11x"``;
    - a ``"gpu"`` token (``"<pkg>_gpu"``) is the same as ``"<pkg>_cuda"``.

    ``<pkg>`` is the part before the first ``_``. A bare flavor (``"cpu"``, ``"cuda"``)
    is not a registry name and raises ``ValueError``; package facades such as
    ``lisatools.get_backend`` add their prefix before calling this function.

    Args:
        backend_name: Registry name or ``"<pkg>_cuda"`` / ``"<pkg>_gpu"`` alias.

    Returns:
        Backend: The loaded backend singleton.

    Raises:
        ValueError: If the (resolved) name is not registered.
        BackendAccessException: If the backend is disabled or failed to load, or, for the
            aliases, if no CUDA flavor is available.

    Example:
        >>> import gpubackendtools
        >>> from gpubackendtools import get_backend, has_backend
        >>> if has_backend("gbt_cpu"):
        ...     xp = get_backend("gbt_cpu").xp
        >>> # GPU machines: get_backend("gbt_cuda") -> first of gbt_cuda13x/12x/11x
    """
    if "cuda" in backend_name.split("_"):
        _base = backend_name.split("_")[0]
        return get_first_backend([_base + "_cuda13x" , _base + "_cuda12x", _base + "_cuda11x"])
    if "gpu" in backend_name.split("_"):
        _base = backend_name.split("_")[0]
        return get_backend(_base + "_cuda")

    return Globals().backends_manager.get_backend(backend_name=backend_name)


def has_backend(backend_name: str) -> bool:
    """
    Test if a backend is available.

    Loads the backend on first call. ``backend_name`` must be a full registry name
    (``"gbt_cuda12x"``); unlike :func:`get_backend`, prefixed ``"<pkg>_cuda"`` /
    ``"<pkg>_gpu"`` aliases are not resolved and raise ``ValueError``.

    Args:
        backend_name: Full registry name.

    Returns:
        bool: ``True`` if the backend loaded, ``False`` if it is disabled or unavailable.

    Raises:
        ValueError: If ``backend_name`` is not a registry name.

    Warning:
        The bare ``"cuda"`` and ``"gpu"`` names are special-cased, but they recurse into
        the bare names ``"cuda11x"`` / ``"cuda12x"`` / ``"cuda13x"``, which are not
        registry names, so they raise ``ValueError`` too. Probe each flavor explicitly,
        e.g. ``any(has_backend(f"gbt_{f}") for f in ("cuda13x", "cuda12x", "cuda11x"))``.
    """
    if backend_name == "cuda":
        return has_backend("cuda11x") or has_backend("cuda12x") or has_backend("cuda13x")
    if backend_name == "gpu":
        return has_backend("cuda")
    return Globals().backends_manager.has_backend(backend_name=backend_name)


def get_first_backend(backend_names: typing.Sequence[str]) -> Backend:
    """Get the first available backend from a list of backend names.

    Entries are full registry names (``"gbt_cuda12x"``) or bare flavors (``"cuda12x"``). A
    bare flavor matches the first registry name, in registration order, ending in
    ``_<flavor>``; since ``gbt_*`` is registered first, bare flavors resolve to
    ``gpubackendtools``' own backends. Pass prefixed names to select another package's.
    See :meth:`~gpubackendtools.gpubackendtools.BackendsManager.get_first_backend`.

    Args:
        backend_names: Candidates in order of preference (non-empty).

    Returns:
        Backend: The first candidate that loads.

    Raises:
        ValueError: If a candidate tried before a loadable one is not registered.
        BackendAccessException: If no candidate is available.

    Example:
        >>> import gpubackendtools
        >>> from gpubackendtools import get_first_backend
        >>> backend = get_first_backend(["gbt_cuda12x", "gbt_cpu"])  # GPU if present
    """
    return Globals().backends_manager.get_first_backend(backend_names)


# TODO check: keep/remove/document -- no callers; raises after import (globals already initialized)
def initialize(*cli_args):
    """Initialize GPUBACKENDTOOLS configuration, logger and backend manager with CLI arguments.

    Calls ``Globals().init(*cli_args)``: the first positional value is taken as the whole
    ``cli_args`` sequence and a second one as ``set_args``. Raises
    :class:`GPUBACKENDTOOLSGlobalsInitializedTwice` unless :func:`reset` was called first.
    """
    Globals().init(*cli_args)


def get_config_setter(reset: bool = False) -> ConfigurationSetter:
    """Get a configuration setter.

    Args:
        reset: Reset already initialized globals first. Default ``False``, which raises
            once ``gpubackendtools`` has been imported. See :class:`ConfigurationSetter`.

    Returns:
        ConfigurationSetter: The globals' setter.

    Raises:
        GPUBACKENDTOOLSGlobalsInitializedTwice: If initialized and ``reset`` is ``False``.
    """
    return Globals().get_configuration_setter(reset=reset)


def reset(quiet: bool = False):
    """Reset global states.

    Logs a warning (unless ``quiet``) and calls :meth:`Globals.reset`; the backend
    registry is empty afterwards until the next ``add_backends`` call.

    Args:
        quiet: Skip the warning. Default ``False``.
    """
    if not quiet:
        get_logger().warning(
            "GPUBACKENDTOOLS globals are about to be reset. Objects built up until now should be deleted to prevent unexpected side-effects."
        )
    Globals().reset()


# Initialize the globals singleton when first importing this file
Globals()


__all__ = [
    "Globals",
    "ConfigurationSetter",
    "get_logger",
    # "get_file_manager",
    "get_config",
    "get_config_setter",
    "get_backend",
    "get_first_backend",
    "has_backend",
    "initialize",
]
