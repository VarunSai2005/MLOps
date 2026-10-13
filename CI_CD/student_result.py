def calculate_result(mark):
    if mark >= 40:
        return "Pass"
    else:
        return "Fail"

if __name__ == "__main__":
    mark = 65
    print("Student Mark:", mark)
    print("Result:", calculate_result(mark))
