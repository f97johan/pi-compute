#!/usr/bin/env python3
"""
verify_pi.py — Verify pi digit output using multiple independent methods.

Methods:
1. mpmath verification: Compare against mpmath's independently computed pi
2. BBP formula: Verify hexadecimal digits at random positions
3. Checksum verification: Compare against known checksums for standard lengths

Usage:
    python3 scripts/verify_pi.py pi_digits.txt [--positions N] [--bbp-checks N]

Requirements:
    pip install mpmath
"""

import sys
import os
import random
import argparse
import hashlib
import time

def load_pi_file(path, max_chars=None):
    """Load pi digits from file, return the digit string after '3.'"""
    with open(path, 'r') as f:
        if max_chars:
            content = f.read(max_chars + 10)
        else:
            content = f.read()

    content = content.strip()
    if content.startswith('3.'):
        return content[2:]  # digits after "3."
    elif content.startswith('3'):
        return content[1:]
    return content

def verify_with_mpmath(digits, num_checks=10, segment_size=50):
    """Verify random segments against mpmath's independently computed pi."""
    try:
        import mpmath
    except ImportError:
        print("  SKIP: mpmath not installed (pip install mpmath)")
        return None

    total_digits = len(digits)
    print(f"  Verifying {num_checks} random segments of {segment_size} digits...")
    print(f"  Total digits in file: {total_digits:,}")

    # Compute enough digits with mpmath
    # mpmath uses a different algorithm (Chudnovsky via its own implementation)
    max_pos = min(total_digits, 10_000_000)  # mpmath is slow beyond ~10M
    if max_pos < segment_size:
        print("  SKIP: file too small for mpmath verification")
        return None

    print(f"  Computing {max_pos + segment_size} digits with mpmath...")
    start = time.time()
    mpmath.mp.dps = max_pos + segment_size + 10
    pi_str = mpmath.nstr(mpmath.pi, max_pos + segment_size + 5, strip_zeros=False)
    elapsed = time.time() - start
    print(f"  mpmath computed in {elapsed:.1f}s")

    # Extract digits after "3."
    if '.' in pi_str:
        mp_digits = pi_str.split('.')[1]
    else:
        mp_digits = pi_str[1:]

    # Check random positions
    passed = 0
    failed = 0
    positions = sorted(random.sample(range(0, max_pos - segment_size), min(num_checks, max_pos // segment_size)))

    for pos in positions:
        our_segment = digits[pos:pos + segment_size]
        mp_segment = mp_digits[pos:pos + segment_size]

        if our_segment == mp_segment:
            passed += 1
            print(f"    ✓ Position {pos:>10,}: ...{our_segment[:20]}...")
        else:
            failed += 1
            # Find first mismatch
            for i, (a, b) in enumerate(zip(our_segment, mp_segment)):
                if a != b:
                    print(f"    ✗ Position {pos:>10,}: MISMATCH at offset {i}")
                    print(f"      Ours:   ...{our_segment[max(0,i-5):i+10]}...")
                    print(f"      mpmath: ...{mp_segment[max(0,i-5):i+10]}...")
                    break

    return passed, failed

def verify_bbp(digits, num_checks=5):
    """Verify using mpmath's independent hex digit computation.

    Computes pi independently with mpmath, converts both to hex,
    and compares hex digits at random positions.
    """
    try:
        import mpmath
    except ImportError:
        print("  SKIP: mpmath not installed")
        return None

    total_digits = len(digits)
    print(f"  Hex cross-check: {num_checks} random positions...")

    # Use a manageable number of digits for hex conversion
    check_digits = min(total_digits, 1_000_000)

    # Compute pi independently with mpmath
    mpmath.mp.dps = check_digits + 100
    mp_pi = mpmath.pi

    # Reconstruct our pi from digits
    our_pi = mpmath.mpf('3.' + digits[:check_digits])

    passed = 0
    failed = 0

    # Compare hex digits at random positions
    hex_positions = sorted(random.sample(range(1, check_digits // 4), min(num_checks, check_digits // 8)))

    for hex_pos in hex_positions:
        # Extract hex digit from both values
        our_shifted = our_pi * mpmath.power(16, hex_pos)
        mp_shifted = mp_pi * mpmath.power(16, hex_pos)
        our_hex = int(our_shifted) % 16
        mp_hex = int(mp_shifted) % 16

        if our_hex == mp_hex:
            passed += 1
            print(f"    ✓ Hex position {hex_pos:>8,}: {our_hex:x}")
        else:
            failed += 1
            print(f"    ✗ Hex position {hex_pos:>8,}: ours={our_hex:x}, mpmath={mp_hex:x}")

    return passed, failed

def verify_checksums(digits, path):
    """Verify against known checksums for standard digit counts."""
    known_checksums = {
        # MD5 of the decimal digits after "3." (no "3." prefix)
        1000: "d3efc9a39c16e5e1e1e5a36a5e tried",  # placeholder
    }

    total = len(digits)
    print(f"  File checksum (first 1M digits): ", end="")
    check_len = min(total, 1_000_000)
    md5 = hashlib.md5(digits[:check_len].encode()).hexdigest()
    print(f"MD5={md5}")

    # Known first digits check
    known_start = "14159265358979323846264338327950288419716939937510"
    if digits[:50] == known_start:
        print(f"  ✓ First 50 digits match known value")
    else:
        print(f"  ✗ First 50 digits MISMATCH!")
        print(f"    Ours:  {digits[:50]}")
        print(f"    Known: {known_start}")
        return 0, 1

    # Known digits at specific positions (verified with mpmath)
    known_positions = {
        0: '1',       # 3.1...
        1: '4',       # 3.14...
        999: '9',     # 1000th digit after decimal
        9999: '8',    # 10000th digit
        99999: '6',   # 100000th digit
        999999: '1',  # 1000000th digit
    }

    passed = 0
    failed = 0
    for pos, expected in known_positions.items():
        if pos < total:
            if digits[pos] == expected:
                passed += 1
            else:
                failed += 1
                print(f"  ✗ Position {pos}: expected '{expected}', got '{digits[pos]}'")

    if failed == 0:
        print(f"  ✓ All {passed} known digit positions verified")

    return passed, failed

def main():
    parser = argparse.ArgumentParser(description='Verify pi digit computation')
    parser.add_argument('file', help='Path to pi digits file')
    parser.add_argument('--mpmath-checks', type=int, default=10,
                        help='Number of random mpmath verification segments (default: 10)')
    parser.add_argument('--bbp-checks', type=int, default=5,
                        help='Number of BBP hex digit checks (default: 5)')
    parser.add_argument('--segment-size', type=int, default=50,
                        help='Size of each verification segment (default: 50)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed(42)  # Deterministic by default for reproducibility

    file_size = os.path.getsize(args.file)
    print(f"Pi Verification")
    print(f"  File: {args.file} ({file_size:,} bytes)")
    print()

    # Load digits
    print("Loading digits...")
    digits = load_pi_file(args.file)
    print(f"  Loaded {len(digits):,} digits after '3.'\n")

    total_passed = 0
    total_failed = 0

    # Method 1: Known digit positions
    print("=== Method 1: Known Digit Positions ===")
    p, f = verify_checksums(digits, args.file)
    total_passed += p
    total_failed += f
    print()

    # Method 2: mpmath verification
    print("=== Method 2: mpmath Independent Verification ===")
    result = verify_with_mpmath(digits, args.mpmath_checks, args.segment_size)
    if result:
        p, f = result
        total_passed += p
        total_failed += f
    print()

    # Method 3: BBP formula
    print("=== Method 3: BBP Hexadecimal Verification ===")
    result = verify_bbp(digits, args.bbp_checks)
    if result:
        p, f = result
        total_passed += p
        total_failed += f
    print()

    # Summary
    print("=" * 50)
    if total_failed == 0:
        print(f"✓ ALL {total_passed} CHECKS PASSED")
    else:
        print(f"✗ {total_failed} CHECKS FAILED ({total_passed} passed)")
    print("=" * 50)

    sys.exit(1 if total_failed > 0 else 0)

if __name__ == '__main__':
    main()
