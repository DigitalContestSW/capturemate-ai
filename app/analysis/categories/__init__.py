from app.analysis.categories import restaurant, schedule, study
from app.analysis.categories.base import CategoryStage2

# 카테고리 -> 2단계 처리 레지스트리.
# 새 카테고리를 추가하려면: categories/<name>.py 를 만들고 여기에 한 줄 등록.
# 등록되지 않은 카테고리(예: life_info, unknown)는 2단계를 건너뛰고
# 1단계 기본 결과만 반환된다.
CATEGORY_STAGE2: dict[str, CategoryStage2] = {
    "schedule": CategoryStage2(schedule.build_schedule_prompt, schedule.ScheduleDetails),
    "study": CategoryStage2(study.build_study_prompt, study.StudyDetails),
    "restaurant": CategoryStage2(restaurant.build_restaurant_prompt, restaurant.RestaurantDetails),
    # "life_info": categories/life_info.py 추가 후 등록
}
