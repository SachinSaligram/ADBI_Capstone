#install.packages("jsonlite")
library(jsonlite)
library(ggplot2)
setwd("/Users/sachin/Google Drive/CSC 591 BI/Projects/Capstone/Yelp Dataset")
tip <- stream_in(file("yelp_academic_dataset_tip.json"))
checkin <- stream_in(file("yelp_academic_dataset_checkin.json"))
user <- stream_in(file("yelp_academic_dataset_user.json"))
review <- stream_in(file("yelp_academic_dataset_review.json"))
business <- stream_in(file("yelp_academic_dataset_business.json"))

tip_df = tbl_df(tip)
checkin_df = tbl_df(checkin)
user_df = tbl_df(user)
review_df = tbl_df(review)
business_df = tbl_df(business)

colnames(tip_df)
# "text"        "date"        "likes"       "business_id" "user_id"     "type"
colnames(checkin_df)
#"time"        "business_id" "type"
colnames(user_df)
#"user_id"            "name"               "review_count"       "yelping_since"     
#"friends"            "useful"             "funny"              "cool"              
#"fans"               "elite"              "average_stars"      "compliment_hot"    
#"compliment_more"    "compliment_profile" "compliment_cute"    "compliment_list"   
#"compliment_note"    "compliment_plain"   "compliment_cool"    "compliment_funny"  
#"compliment_writer"  "compliment_photos"  "type"
colnames(review_df)
#"review_id"   "user_id"     "business_id" "stars"       "date"        "text"       
#"useful"      "funny"       "cool"        "type"
colnames(business_df)
#"business_id"  "name"         "neighborhood" "address"      "city"         "state"       
#"postal_code"  "latitude"     "longitude"    "stars"        "review_count" "is_open"     
#"attributes"   "categories"   "hours"        "type"

user_bus = distinct(bind_rows(tip_df[c("user_id","business_id")],review_df[c("user_id","business_id")]))
merge_bus = inner_join(x = user_bus, y = business_df[c("business_id","stars")],by = "business_id")
merge_bus_user = inner_join(x = merge_bus, y = user_df[c("user_id","review_count","average_stars")], by = "user_id")

names(merge_bus_user) = c("UserID", "BusinessID", "StarofBusiness", "UserReviewCount", "UserAvgStars")

ggplot(merge_bus_user, aes(x = UserAvgStars)) + geom_histogram(alpha = .50, binwidth=.1, colour = "black")
#ggplot(merge_bus_user, aes(UserAvgStars, UserReviewCount)) + geom_point()

# Predict how many star a user will rate a business based on the user’s rating behaviour
fit1 <- lm(StarofBusiness ~ UserAvgStars, data = merge_bus_user)
fit2 <- lm(StarofBusiness ~ UserReviewCount, data = merge_bus_user)
fit3 <- lm(StarofBusiness ~ UserAvgStars+UserReviewCount, data = merge_bus_user)
anova(fit1, fit2, fit3)

inTrain = createDataPartition(merge_bus_user$StarofBusiness, p = 0.5, list=FALSE)
training = merge_bus_user[ inTrain,]
# 2326007       5
testing = merge_bus_user[-inTrain,]
# 2326007       5
dim(training)

modFit <- train(StarofBusiness ~ UserAvgStars + UserReviewCount,data=training,method="lm")
predictions <- round(predict(modFit, testing))
u = union(predictions, testing$StarofBusiness)
t = table(factor(predictions, u), factor(testing$StarofBusiness, u))
confusionMatrix(t)

difference <- testing$StarofBusiness - predictions
hist(difference, breaks = 10, main = "Difference between Predicted and Actual")

difference = as.data.frame(table(difference))
(difference[4,2]+difference[5,2]+difference[6,2]+difference[7,2]+difference[8,2])/sum(difference[,2])


