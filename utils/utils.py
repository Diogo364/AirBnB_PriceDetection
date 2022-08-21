def parse_currency_str_to_float(in_str):
    return float(in_str.replace('$', '').replace(',', '.'))