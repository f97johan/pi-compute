#!/usr/bin/env python3
"""
Fetch reference pi digits from pi.delivery API and save to files.
Also computes SHA-256 checksums for verification.

Usage:
    python fetch_reference.py [--digits N] [--output FILE]

Examples:
    python fetch_reference.py --digits 10000 --output pi_10000.txt
    python fetch_reference.py --digits 1000000 --output pi_1000000.txt
"""

import argparse
import hashlib
import sys

def fetch_from_mpmath(digits: int) -> str:
    """Compute pi using mpmath (independent verification source)."""
    try:
        import mpmath
        mpmath.mp.dps = digits + 10  # Extra precision
        pi_str = mpmath.nstr(mpmath.pi, digits + 1, strip_zeros=False)
        # Ensure we have exactly the right number of digits after decimal point
        dot_pos = pi_str.find('.')
        if dot_pos >= 0:
            pi_str = pi_str[:dot_pos + 1 + digits]
        return pi_str
    except ImportError:
        print("mpmath not installed. Install with: pip install mpmath", file=sys.stderr)
        sys.exit(1)

def compute_checksum(pi_str: str) -> str:
    """Compute SHA-256 checksum of a pi digit string."""
    return hashlib.sha256(pi_str.encode('utf-8')).hexdigest()

def main():
    parser = argparse.ArgumentParser(description='Fetch reference pi digits')
    parser.add_argument('--digits', type=int, default=10000,
                        help='Number of decimal digits (default: 10000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: pi_<digits>.txt)')
    parser.add_argument('--checksum', action='store_true',
                        help='Print SHA-256 checksum')
    args = parser.parse_args()

    if args.output is None:
        args.output = f'pi_{args.digits}.txt'

    print(f"Computing {args.digits} digits of pi using mpmath...")
    pi_str = fetch_from_mpmath(args.digits)

    with open(args.output, 'w') as f:
        f.write(pi_str)

    print(f"Written to {args.output}")
    print(f"Length: {len(pi_str)} characters")
    print(f"Preview: {pi_str[:80]}...")

    if args.checksum:
        checksum = compute_checksum(pi_str)
        print(f"SHA-256: {checksum}")

if __name__ == '__main__':
    main()
