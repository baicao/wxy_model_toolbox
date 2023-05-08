from common.config_tools import SimpleParser


def parse_es_func(conf_file: str) -> dict:
    parser = SimpleParser(conf_file)

    def split_func(x):
        return x.split(",")

    result = {}
    for key in parser.keys:
        es_hosts = parser.get_field_value(key,
                                          "host",
                                          transfer_function=split_func)
        es_ports = parser.get_field_value(key,
                                          "port",
                                          transfer_function=split_func)

        # invalid host
        if len(es_hosts) == 0 or len(
                es_ports) == 0 or len(es_hosts) < len(es_ports):
            raise ValueError("please specify valid host/port")

        # size not match
        if len(es_hosts) > 1 and len(es_ports) > 1 and len(es_hosts) != len(
                es_ports):
            raise ValueError(
                "sizeof hosts should match sizeof ports,but you give {} & {}".
                format(len(es_hosts), len(es_ports)))

        # if only 1 port,tie
        if len(es_ports) == 1:
            port = es_ports[0]
            es_conn_dict = [{"host": item, "port": port} for item in es_hosts]
        else:
            es_conn_dict = [{
                "host": host,
                "port": port
            } for host, port in zip(es_hosts, es_ports)]

        index_format = parser.get_field_value(key,
                                              "index_format",
                                              transfer_function=None)
        doc_type = parser.get_field_value(key,
                                          "doc_type",
                                          transfer_function=None)
        result[key] = {
            "hosts": es_conn_dict,
            "index_format": index_format,
            "doc_type": doc_type,
        }
    return result
