# AFC Configuration Interface

Part of the [Advanced Facade Controller (AFC)](../../README.md) project.

A Flask-based web interface for generating the AFC configuration file (`user_config.json`).

This interface only creates and stores the configuration for AFC. To run the AFC controller, use the [FMLC web interface](https://github.com/LBNL-ETA/FMLC/tree/master#web-interface).

## Getting started

```bash
python3 afc/interface/server.py
```

Then open `http://127.0.0.1:8000` in a browser, fill in the configuration, and click **Save Configuration**. The resulting `user_config.json` is written next to `server.py` and is picked up automatically by the AFC controller.

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind |
| `--port` | `8000` | Port to listen on |
| `--config-dir PATH` | script directory | Folder to read/write `user_config.json` |
| `--config-file PATH` | — | Path to an existing JSON config file to use as the active config |

`--config-dir` and `--config-file` are mutually exclusive. When neither is provided, `user_config.json` is read from and written to the same directory as `server.py`.

## Docker example

```bash
set username=%USERNAME%
set container=cgehbauer/jupyter_radiance_eplus:v4
docker run -it -p 127.0.0.1:8000:8000 -v C:\Users\%username%:/home/%username% --rm %container%
```

Then inside the container:

```bash
python3 afc/interface/server.py --host 0.0.0.0 --port 8000
```

Then open: `http://127.0.0.1:8000`

## Pages

| URL | Description |
|-----|-------------|
| `/` | Occupant preferences — brightness and glare sliders |
| `/config/` | System configuration — site, location, room dimensions, occupants, windows, HVAC, lighting, and tariff |
| `/debug/` | View all stored settings or reset the configuration |
