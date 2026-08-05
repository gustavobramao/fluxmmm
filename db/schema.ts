import { integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const datasets = sqliteTable("datasets", {
  hash: text("hash").primaryKey(),
  name: text("name").notNull(),
  objectKey: text("object_key").notNull(),
  rowCount: integer("row_count").notNull(),
  columnCount: integer("column_count").notNull(),
  createdAt: text("created_at").notNull(),
});

export const modelRuns = sqliteTable("model_runs", {
  fingerprint: text("fingerprint").primaryKey(),
  datasetHash: text("dataset_hash").notNull(),
  kind: text("kind").notNull(),
  resultJson: text("result_json").notNull(),
  createdAt: text("created_at").notNull(),
});
