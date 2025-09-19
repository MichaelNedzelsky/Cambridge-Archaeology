#!/usr/bin/env python3
import pandas as pd
import sys

def clean_dataset(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Remove records with Arbury in Project ID
    df = df[~df['Project ID'].str.contains('Arbury', na=False)]

    # Remove records where Adult column is 'N'
    df = df[df['Adult (Y/N/unknown)'].str.strip() != 'N']

    # Update Site ID column with simplified site names
    def map_site(site_id):
        if pd.isna(site_id):
            return None
        site_id = str(site_id).lower()
        if 'knobbs' in site_id:
            return 'Knobbs'
        elif 'nw_cambridge' in site_id or 'northwest' in site_id:
            return 'NW_Cambridge'
        elif 'vicar' in site_id:
            return 'Vicar_Farm'
        elif 'fenstanton' in site_id:
            return 'Fenstanton'
        elif 'duxford' in site_id:
            return 'Duxford'
        else:
            return None

    df['Site ID'] = df['Site ID'].apply(map_site)

    # Remove Sex determination column
    df = df.drop('Sex determination', axis=1)

    # Convert Sample ID for kin to list format
    def parse_kin_ids(kin_str):
        if pd.isna(kin_str) or kin_str == '':
            return '[]'
        # Handle multi-line entries and clean up
        kin_str = str(kin_str).replace('\n', ' ')
        # Split by spaces and filter out empty strings
        kin_ids = [id.strip() for id in kin_str.split() if id.strip()]
        # Format as Python list string
        return str(kin_ids)

    df['Sample ID for kin'] = df['Sample ID for kin'].apply(parse_kin_ids)

    # Output to stdout
    df.to_csv(sys.stdout, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_dataset.py <input_csv_file>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    clean_dataset(input_file)