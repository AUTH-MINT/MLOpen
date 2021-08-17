def list_to_js_obj(lst):
    ret = []
    for item in lst:
        ret.append({'name': item})
    return ret
