import os
import threading
import sqlalchemy
from sqlalchemy import event

import google.auth
from google.auth.transport.requests import Request
from google.cloud.sql.connector import Connector

class CloudSQLPostgresConnector:
  _instance: "CloudSQLPostgresConnector" = None
  _iam_creds = None
  _lock = threading.Lock()

  def __new__(cls, instance_name, user, password, database="postgres", driver="pg8000") -> "CloudSQLPostgresConnector":
    with cls._lock:
      if cls._instance is None:
        cls._instance = super().__new__(cls)
        cls._iam_creds, _ = google.auth.default(
          scopes=["https://www.googleapis.com/auth/sqlservice.login"]
        )
        cls._instance._instance_name = instance_name
        cls._instance._user = user
        cls._instance._password = password
        cls._instance._database = database
        cls._instance._driver = driver
        cls._instance._connector = None
        if driver != "psycopg":
          cls._instance._connector = Connector()
        cls._instance._engine = cls.__create_engine(driver)

    return cls._instance
  
  def get_driver(self):
    return self._instance._driver

  def get_engine(self) -> sqlalchemy.engine.base.Engine:
    if self._instance._engine is None:
      raise RuntimeError("The engine has been disposed. Please recreate it.")
    return self._instance._engine
  
  def close(self):
    if self._instance._engine:
      self._instance._engine.dispose()
      self._instance._engine = None

    if self._instance._connector:
      self._instance._connector.close()
      self._instance._connector = None
  
  def connect(self):
    with self._lock:
      driver = self._instance._driver
      if driver != "psycopg":
        self._instance._connector = Connector()
      self._instance._engine = self.__create_engine(driver)

  @classmethod
  def __auto_iam_authn(cls, **kwargs):
    if not cls._iam_creds.valid:
      request = Request()
      cls._iam_creds.refresh(request)

    kwargs["cparams"]["password"] = str(cls._iam_creds.token)

  @classmethod
  def __getconn(cls, **kwargs):
    if cls._instance._connector is None:
      raise RuntimeError("Connector is not set")
    return cls._instance._connector.connect(
      instance_connection_string=cls._instance._instance_name,
      driver=cls._instance._driver,
      user=cls._instance._user,
      password=cls._instance._password,
      db=cls._instance._database
    )
  
  @classmethod
  def __create_engine(cls, driver) -> sqlalchemy.engine.base.Engine:
    conn_uri = sqlalchemy.engine.url.URL(
      drivername=f"postgresql+{driver}",
      username=cls._instance._user,
      password=cls._instance._password,
      host=None,
      port=None,
      database=cls._instance._database,
      query={
        "host": f"{os.getenv('CLOUD_SQL_UNIX_SOCKET_ROOT', '/cloudsql')}/{cls._instance._instance_name}"
      },
    )
    engine = sqlalchemy.create_engine(
      url=conn_uri,
      pool_pre_ping=True
    )
    if driver == "psycopg":
      event.listen(engine, "do_connect", cls.__auto_iam_authn, named=True)
    else:
      event.listen(engine, "do_connect", cls.__getconn, named=True)
    return engine