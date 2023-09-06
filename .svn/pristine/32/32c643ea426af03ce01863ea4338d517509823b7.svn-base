#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：adi-datatool
@File    ：test.py
@Author  ：LvYong
@Date    ：2022/3/3 10:50 
"""
import pymysql
import re


if __name__ == "__main__":
    # database connection
    connection = pymysql.connect(host="localhost", user="root", passwd="123", database="demo")
    cursor = connection.cursor()

    # 查询已存在的表
    cursor.execute("SHOW TABLES;")
    tables = [cursor.fetchall()]
    table_list = re.findall('(\'.*?\')', str(tables))
    table_list = [re.sub("'", '', each) for each in table_list]

    if 'artists'.lower() not in table_list:
        # creating table
        ArtistTableSql = """CREATE TABLE artists(
                            ID INT(20) PRIMARY KEY AUTO_INCREMENT,
                            NAME  CHAR(20) NOT NULL,
                            TRACK CHAR(10))"""

        cursor.execute(ArtistTableSql)

    # queries for inserting values
    insert1 = "INSERT INTO demo.artists(NAME, TRACK) VALUES('Towang', 'Jazz' );"
    insert2 = "INSERT INTO demo.artists(NAME, TRACK) VALUES('Sadduz', 'Rock' );"
    cursor.execute(insert1)
    cursor.execute(insert2)
    connection.commit()

    # queries for retrievint all rows
    retrive = "Select * from demo.artists;"
    cursor.execute(retrive)
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    # update
    updateSql = f"UPDATE  demo.artists SET NAME= 'Tauwang'  WHERE ID = '{rows[0][0]}';"
    cursor.execute(updateSql)
    connection.commit()

    # delete
    deleteSql = f"DELETE FROM demo.artists WHERE ID = '{rows[1][0]}'; "
    cursor.execute(deleteSql)
    connection.commit()

    # # delete table
    # dropSql = "DROP TABLE IF EXISTS artists;"
    # cursor.execute(dropSql)
    # connection.commit()

    # close.
    cursor.close()
    connection.close()
