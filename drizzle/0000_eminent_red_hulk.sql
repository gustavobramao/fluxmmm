CREATE TABLE `datasets` (
	`hash` text PRIMARY KEY NOT NULL,
	`name` text NOT NULL,
	`object_key` text NOT NULL,
	`row_count` integer NOT NULL,
	`column_count` integer NOT NULL,
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `model_runs` (
	`fingerprint` text PRIMARY KEY NOT NULL,
	`dataset_hash` text NOT NULL,
	`kind` text NOT NULL,
	`result_json` text NOT NULL,
	`created_at` text NOT NULL
);
