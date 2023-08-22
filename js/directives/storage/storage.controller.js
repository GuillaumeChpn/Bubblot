(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('storageCtrl', storageCtrl);

    function storageCtrl($scope, $element) {
        var vm = this;
        var cadran=$element.find("cadran").context;
        cadran.addEventListener("wheel", scrollPumpPower, false);
        function scrollPumpPower(event) {
            if($scope.focusIndex==4){
                if($scope.pumpPower>=0 && $scope.pumpPower<=1){
                    $scope.pumpPower=$scope.pumpPower + event.deltaY/5000;
                    if($scope.pumpPower>1){
                        $scope.pumpPower=1;
                    }
                    else if($scope.pumpPower<0){
                        $scope.pumpPower=0;
                    }
                    $scope.$apply();
                }  
            }
        }
        $scope.initTimer  = function () {  
            $scope.pumpOn=!$scope.pumpOn;
            //$scope.filledOk=!$scope.filledOk; 
            //setTimeout($scope.filled, 25000);
        }
        $scope.filled  = function () { 
            $scope.pumpOn=false;  
            $scope.filledOk=true;        
        }
    }

}());
