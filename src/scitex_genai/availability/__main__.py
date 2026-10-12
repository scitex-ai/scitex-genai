"""Probe one credential without putting its value into process arguments."""

import argparse
import json
import os

from . import probe_provider_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--endpoint-url")
    parser.add_argument("--protocol")
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    token = os.environ.get(args.key_env)
    if not token:
        parser.error("The selected key environment variable is empty or absent")
    try:
        result = probe_provider_key(
            args.provider,
            args.model,
            token,
            endpoint_url=args.endpoint_url,
            protocol=args.protocol,
            timeout_s=args.timeout,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result.to_dict()))
    return 0 if result.available is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
