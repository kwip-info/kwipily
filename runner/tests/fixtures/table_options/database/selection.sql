CREATE TABLE samples (id INTEGER, cohort TEXT, name TEXT, value REAL);
INSERT INTO samples VALUES (4,'B','delta',4.5),(2,'A','beta',2.5),(1,'A','alpha',1.5),(3,'B','gamma',NULL);
CREATE TABLE alternative (id INTEGER, cohort TEXT, name TEXT, value REAL);
INSERT INTO alternative VALUES (99,'Z','other-table',99.5);
CREATE TABLE binary_values (id INTEGER, payload BLOB);
INSERT INTO binary_values VALUES (1,X'00FF41');
CREATE TABLE text_dates (id INTEGER, moment DATE, amount NUMERIC);
INSERT INTO text_dates VALUES (1,'2026-10-13',1.25);
