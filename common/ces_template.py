from datetime import datetime
from elasticsearch import helpers, Elasticsearch


class CESTemplate(object):

    def __init__(
        self,
        host,
        port,
        index,
        doc_type,
        index_format_date=None,
        user=None,
        password=None,
        timeout=3,
        max_retries=10,
    ):
        hosts = host.split(",")
        ports = port.split(",")
        ip_list = [{"host": h, "port": p} for h, p in zip(hosts, ports)]

        self.index = index
        if index_format_date is not None:
            now = datetime.now()
            self.index = index.format(now.strftime(index_format_date))
        self.doc_type = doc_type
        self.timeout = timeout
        self.max_retries = max_retries
        if user is not None and password is not None:
            self.http_auth = (user, password)
            self.client = Elasticsearch(
                ip_list,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_on_timeout=True,
                http_auth=self.http_auth,
            )
        else:
            self.client = Elasticsearch(
                ip_list,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_on_timeout=True,
            )

    def import_data_batch(self, id_, packages, batch_size):
        """ 批量插入数据
        :param id_:
        :param packages:
        :param batch_size:
        :return:
        """
        actions = []
        for d in packages:
            actions.append({
                "_op_type": "update",
                "_index": self.index,
                "_type": self.doc_type,
                "_id": d[id_],
                "_source": {
                    "doc": d,
                    "doc_as_upsert": True
                },
            })
            if len(actions) >= batch_size:
                helpers.bulk(self.client, actions)
                actions = []

        if len(actions) > 0:
            helpers.bulk(self.client, actions)

    def init_scroll(self, body, size, timeout="1m"):
        """标准滚动
        :param body:
        :param timeout:
        :param size:每次请求的返回结果
        :return:scroll_id, total_size, hits
        """
        page = self.client.search(body=body,
                                  index=self.index,
                                  params={
                                      "scroll_size": size,
                                      "scroll": timeout
                                  })
        scroll_id = page["_scroll_id"]
        total_size = page["hits"]["total"]

        hits = page["hits"]["hits"]

        return scroll_id, total_size, hits

    def search_scroll(self, scroll_id, scroll="2m"):
        """通过scroll_id 查询数据
        :param scroll_id:
        :return: scroll_id, hits
        """
        page = self.client.scroll(
            scroll_id=scroll_id,
            params={"scroll": scroll},
        )
        scroll_id = page["_scroll_id"]
        hits = page["hits"]["hits"]
        return scroll_id, hits

    def search(self, body, size, timeout="1m"):
        page = self.client.search(index=self.index,
                                  params={
                                      "size": size,
                                      "scroll": timeout
                                  },
                                  body=body)
        scroll_id = page["_scroll_id"]
        total_size = page["hits"]["total"]
        data = page["hits"]["hits"]
        data_list = []
        while len(data) > 0:
            data_list.extend(data)
            scroll_id, data = self.search_scroll(scroll_id, scroll=timeout)
        return data_list, total_size

    def aggregations(self, body, timeout="1m", size=0):
        page = self.client.search(index=self.index,
                                  body=body,
                                  params={
                                      "size": size,
                                      "scroll": timeout
                                  })
        total_size = page["hits"]["total"]
        agg = page["aggregations"]
        return agg, total_size
