"""
Prompt templates for the three complexity tiers.

Each template returns a string with placeholders {schema} and {question}.
Designed for instruct-tuned 7B-14B models. Keep these stable across runs
so token counts are comparable.
"""

SIMPLE_PROMPT = """You are an expert SQL writer. Generate a single valid SQLite SQL query that answers the question. Use only the tables and columns shown.

### Schema
{schema}

### Question
{question}

### Output
Return only the SQL query, with no explanation, no markdown fences, no leading SELECT label.

SQL:"""


MEDIUM_PROMPT = """You are an expert SQL writer. Generate a single valid SQLite SQL query that answers the question. Use only the tables and columns shown. Think step by step, then output only the SQL.

### Schema (with foreign keys)
{schema}

### Example
Question: How many students are in each department?
Reasoning: We need to count students grouped by department. Join students to departments on dept_id, group by department name, count.
SQL: SELECT d.name, COUNT(s.id) FROM students s JOIN departments d ON s.dept_id = d.id GROUP BY d.name;

### Question
{question}

### Output
First reason briefly, then on a new line write `SQL:` followed by the query on a single line. Do not use markdown fences.
"""


COMPLEX_PROMPT = """You are an expert SQL writer. Generate a single valid SQLite SQL query that answers the question. Decompose the problem into sub-steps, then write the final SQL. Use only the tables and columns shown.

### Schema (with foreign keys and sample rows)
{schema}

### Example 1
Question: Find the top 3 departments by average student grade, excluding departments with fewer than 5 students.
Decomposition:
  1. Join students, enrollments, and departments.
  2. Group by department, compute average grade and student count.
  3. Filter groups with HAVING count >= 5.
  4. Order by average grade descending, limit 3.
SQL: SELECT d.name, AVG(e.grade) AS avg_grade FROM departments d JOIN students s ON s.dept_id = d.id JOIN enrollments e ON e.student_id = s.id GROUP BY d.name HAVING COUNT(DISTINCT s.id) >= 5 ORDER BY avg_grade DESC LIMIT 3;

### Example 2
Question: Among courses with more than 50 enrollments, which has the lowest average grade?
Decomposition:
  1. Aggregate enrollments per course.
  2. Filter to courses with count > 50 using HAVING.
  3. Compute average grade per course.
  4. Order ascending and take first.
SQL: SELECT c.title, AVG(e.grade) AS avg_grade FROM courses c JOIN enrollments e ON e.course_id = c.id GROUP BY c.id, c.title HAVING COUNT(*) > 50 ORDER BY avg_grade ASC LIMIT 1;

### Question
{question}

### Output
First write the decomposition as numbered steps, then on a new line write `SQL:` followed by the final query on a single line. Do not use markdown fences.
"""


PROMPTS = {
    "SIMPLE": SIMPLE_PROMPT,
    "MEDIUM": MEDIUM_PROMPT,
    "COMPLEX": COMPLEX_PROMPT,
}
