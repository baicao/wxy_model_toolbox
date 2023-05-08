import traceback
import math
import json
import pymysql
from pymysql.cursors import DictCursor


class DBTemplate(object):
    __conn = None
    __id = 0
    __affected_rows = 0

    def __init__(
        self,
        host=None,
        user=None,
        password=None,
        db=None,
        port: int = 3306,
        debug=False,
        connect_timeout=10,
        read_timeout=60,
    ):

        self.host = host
        self.username = user
        self.password = password
        self.database = db
        self.port = port
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.debug = debug
        print(self.__repr__())
        if host:
            self.__conn = self._connect()

    def __repr__(self):
        return json.dumps({
            "host": self.host,
            "username": self.username,
            "password": self.password,
            "database": self.database,
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
        })

    def address_init(
        self,
        host=None,
        username=None,
        password=None,
        database=None,
        port: int = 3306,
    ):
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.port = port

        self.__conn = self._connect()

    def _connect(self):
        connect = pymysql.connect(
            host=self.host,
            user=self.username,
            password=self.password,
            database=self.database,
            port=self.port,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            charset="utf8",
        )
        return connect

    # 插入记录,成功返回ID，失败返回0
    def insert(self, table, data):
        columns = []
        values = []
        placeholder = []
        for column in data.keys():
            columns.append(column)
            values.append(data[column])
            placeholder.append("%s")

        column_names = "`" + ("`,`".join(columns)) + "`"
        search_values = ",".join(placeholder)
        sql = f"INSERT INTO {table} ({column_names}) VALUES ({search_values})"

        flag = self.execute(sql, values)

        if flag:
            return self.__id
        return 0

    # 批量插入记录
    def insert_batch(self, table, data):
        data_list = []
        if isinstance(data, list):
            data_list = data
        elif isinstance(data, dict):
            data_list.append(data)

        placeholder = []
        columns = []
        values = []
        for k, _ in enumerate(data_list):
            rows = data_list[k]
            vals = []
            for column in rows.keys():
                if k == 0:
                    columns.append(column)
                    placeholder.append("%s")
                vals.append(rows[column])
            if len(vals) > 0:
                values.append(vals)

        sql = ("INSERT INTO `" + table + "` (`" + "`,`".join(columns) +
               "`) values (" + ",".join(placeholder) + ")")
        flag = self.execute(sql, values, True)
        if flag:
            return self.__id
        return 0

    # 更新记录
    def update(self, table, data, where):
        sets = []
        values = []
        for column in data.keys():
            sets.append(column + "=%s")
            values.append(data[column])

        sql = "UPDATE `" + table + "` SET " + ",".join(sets)

        if isinstance(where, str):
            sql += f" WHERE {where}"
        elif isinstance(where, dict):
            where_list = []
            for column in where.keys():
                where_list.append(column + "=%s")
                values.append(where[column])
            sql += " WHERE " + ",".join(where_list)
        else:
            print("where 参数格式异常：", where)
            return 0
        flag = self.execute(sql, values)
        if flag:
            return self.__affected_rows
        return 0

    # 删除记录
    def delete(self, table, where):
        values = []
        sql = "DELETE FROM `" + table + "`"
        if isinstance(where, str):
            sql += f" WHERE {where}"
        elif isinstance(where, dict):
            sets = []
            for column in where.keys():
                sets.append(column + "=%s")
                values.append(where[column])
            sql += " WHERE " + ",".join(sets)
        else:
            return 0

        flag = self.execute(sql, values)
        if flag:
            return self.__affected_rows
        return 0

    # 有则更新，无则插入记录
    def create_or_replace(self, table, data):
        data_list = []
        if isinstance(data, list):
            data_list = data
        elif isinstance(data, dict):
            data_list.append(data)

        placeholder = []
        columns = []
        values = []
        update_columns = []
        for k, _ in enumerate(data_list):
            rows = data_list[k]
            vals = []
            for column in rows.keys():
                if k == 0:
                    update_columns.append("`" + column + "`=VALUES(" + column +
                                          ")")
                    columns.append(column)
                    placeholder.append("%s")
                vals.append(rows[column])
            if len(vals) > 0:
                values.append(vals)
        columns_name = "`,`".join(columns)
        search_value = ",".join(placeholder)
        sql = f"INSERT IGNORE INTO `{table}` (`{columns_name}`) values ({search_value})"
        sql += " ON DUPLICATE KEY UPDATE " + ",".join(update_columns)
        flag = self.execute(sql, values, True)
        if flag:
            return self.__affected_rows
        return 0

    """
    封装查询条件，兼容两种格式
    1、{'id':1,'id':2}
    2、[['id','=',1],['id','=',2]
    """

    @staticmethod
    def package_where(where=None):
        if isinstance(where, dict):
            place = []
            for k, v in where.items():
                place.append(k + " = " + str(v))
            place = " AND ".join(place)
        elif isinstance(where, list):
            place = []
            for k in where:
                if len(k) == 3:
                    v = k[2]
                    if k[1] == "in":
                        v = " (" + ",".join(v) + ")"
                    place.append(k[0] + " " + k[1] + " " + v)
            place = " AND ".join(place)
        else:
            place = where
        return place

    # 查询所有记录，防止内存溢出限制最大条数200000
    def fetch_all(self,
                  table,
                  where=None,
                  columns=None,
                  offset=0,
                  limit=200000,
                  orderby=None):

        if not columns:
            columns = ["*"]
        if isinstance(columns, list) or isinstance(columns, tuple):
            columns = ",".join(columns)

        w = self.package_where(where)

        sql = "SELECT " + columns + " FROM " + table + " WHERE 1=1"
        if w:
            sql += " AND " + w
        if orderby:
            sql += " ORDER BY " + orderby

        if limit > 200000:
            limit = 200000
        if offset > 0:
            sql += " LIMIT " + str(offset) + "," + str(limit)
        else:
            sql += " LIMIT " + str(limit)
        print(sql)
        data = self.query(sql)
        return data

    def fetch_all_batch(self, table, where=None, columns=None, limit=10000):
        """
        查询所有记录，分批返回
        :param where:
        :param columns:
        :param limit:
        :param order_by:
        :return:
        """
        if columns is None:
            columns = list()
        if where is None:
            where = list()
        if limit > 50000:
            limit = 50000

        page = 0
        total = self.count(table, where)
        pages = math.ceil(total / limit)

        while True:
            data_list = {
                "total": total,
                "items": [],
                "pages": pages,
                "per_page": limit,
                "current_page": page,
            }

            # 查询 page 从0开始（前端页面从1开始）
            offset = page * limit

            data_list["items"] = self.fetch_all(table, where, columns, offset,
                                                limit)
            yield data_list

            page += 1
            if page > pages:
                break

    # 查询一条记录
    def fetch_one(self, table, where=None, columns=None):
        if columns is None:
            columns = list()
        if where is None:
            where = list()
        data = self.fetch_all(table, where, columns, 0, 1)
        if len(data) > 0:
            return data[0]
        return []

    # 查询记录数
    def count(self, table, where):
        w = self.package_where(where)
        sql = "SELECT count(*) AS total FROM `" + table + "` " + w

        data = self.query(sql)
        return data[0]["total"]

    # 分页查询记录(page从1开始)
    def pagination(self, table, where=None, columns=None, page=1, limit=50000):
        if columns is None:
            columns = list()
        if where is None:
            where = list()
        data_list = {
            "total": 0,
            "items": [],
            "pages": 0,
            "per_page": limit,
            "current_page": page,
        }
        # 查询 page 从0开始（前端页面从1开始）
        page -= 1
        if page < 0:
            page = 0
        if limit > 50000:
            limit = 50000
        offset = page * limit
        data_list["items"] = self.fetch_all(table, where, columns, offset,
                                            limit)
        data_list["total"] = self.count(table, where)
        if limit > 0:
            data_list["pages"] = math.ceil(data_list["total"] / limit)
        return data_list

    # 执行select查询语句
    def query(self, sql):
        if not self.__conn:
            return False
        cursor = self.__conn.cursor(DictCursor)
        try:
            cursor.execute(sql)
            if self.debug:
                print(sql)

            data = cursor.fetchall()
            return data
        except:  # pylint: disable=bare-except
            traceback.print_exc()
        return []

    # 执行insert、update、delete、alter等操作语句
    def execute(self, sql, values=None, is_many=False):
        if values is None:
            values = list()
        if not self.__conn:
            return False
        cursor = self.__conn.cursor()
        try:
            if is_many:
                cursor.executemany(sql, values)
            else:
                cursor.execute(sql, values)
            self.__affected_rows = cursor.rowcount
            self.__id = self.__conn.insert_id()
            if self.debug:
                print(sql, "%", tuple(values))
            self.__conn.commit()
            return True
        except:  # pylint: disable=bare-except
            self.__conn.rollback()
            print("error sql:", sql, "%", tuple(values))
            traceback.print_exc()
        return False

    # 关闭数据库连接
    # 销毁对象时关闭数据库连接
    def close(self):
        if self.__conn:
            try:
                self.__conn.close()
                self.__conn.cursor()
            except:  # pylint: disable=bare-except
                traceback.print_exc()
