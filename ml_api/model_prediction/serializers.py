from rest_framework import serializers

class PredictionInputSerializer(serializers.Serializer):
    Store = serializers.IntegerField()
    Open = serializers.IntegerField()
    Promo = serializers.IntegerField()
    StateHoliday = serializers.CharField(max_length=1)
    SchoolHoliday = serializers.IntegerField()
    StoreType = serializers.CharField(max_length=1)
    Assortment = serializers.CharField(max_length=1)
    CompetitionDistance = serializers.FloatField()
    Promo2 = serializers.IntegerField()
    Date = serializers.DateField()
