from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'krishna'
    age: Optional[int] = None
    email: EmailStr
    cgpa:float = Field(gt=0, lt=10, default=8.78, description='A decimal value representing grade of the student.')

new_student = {'age': '22', 'email': 'k@gmail.com', 'cgpa': 9.5}

student = Student(**new_student)

print(student)

student_dict = dict(student)

print(student_dict)

student_json = student.model_dump_json()

print(student_json)