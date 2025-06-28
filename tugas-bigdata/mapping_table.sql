-- default.jabatan definition

CREATE TABLE `default`.`jabatan`(
  `id` int, 
  `nama_jabatan` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
LOCATION
  'file:/opt/hive/data/warehouse/jabatan'
TBLPROPERTIES (
  'bucketing_version'='2', 
  'last_modified_by'='hive', 
  'last_modified_time'='1751125052', 
  'transactional'='true', 
  'transactional_properties'='default', 
  'transient_lastDdlTime'='1751125110');


-- default.pegawai definition

CREATE TABLE `default`.`pegawai`(
  `id` int, 
  `nama` string, 
  `jabatan_id` int)
CLUSTERED BY (
  `id`)
INTO 4 BUCKETS
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
LOCATION
  'file:/opt/hive/data/warehouse/pegawai'
TBLPROPERTIES (
  'bucketing_version'='2', 
  'transactional'='true', 
  'transactional_properties'='default', 
  'transient_lastDdlTime'='1751126308');