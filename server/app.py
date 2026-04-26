from faultline_env.server.app import app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    from faultline_env.server.app import main as run_app

    run_app(host=host, port=port)


if __name__ == "__main__":
    main()

__all__ = ["app", "main"]
