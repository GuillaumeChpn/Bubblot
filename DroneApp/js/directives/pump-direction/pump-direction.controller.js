(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('pumpDirectionCtrl', pumpDirectionCtrl);

    function pumpDirectionCtrl($scope) {
        var vm = this;
        var cadran=document.getElementsByClassName('cadrans-container direction');
        for(var i=2; i<cadran.length; i++){
            cadran[i].addEventListener("wheel", scrollDirection, false);
        }
        function scrollDirection(event) {
            if($scope.focusIndex==1){
                var clientRect = this.getBoundingClientRect();
                var centerX = clientRect.left + (clientRect.width/2);
                if(event.screenX < centerX){
                    if($scope.isCtrl){
                        $scope.angleNorth=$scope.angleNorth+event.deltaY/100;
                    }
                    else{
                        $scope.angleNorth=$scope.angleNorth+event.deltaY/20;
                    }
                    if($scope.angleNorth > 360){
                        $scope.angleNorth=0;
                    }
                    else if($scope.angleNorth < 0){
                        $scope.angleNorth=360;
                    }
                }
                else{
                    if($scope.isCtrl){
                        $scope.angleStorage=$scope.angleStorage+event.deltaY/100;
                    }
                    else{
                        $scope.angleStorage=$scope.angleStorage+event.deltaY/20;
                    }
                    if($scope.angleStorage > 360){
                        $scope.angleStorage=0;
                    }
                    else if($scope.angleStorage < 0){
                        $scope.angleStorage=360;
                    }  
                }
                $scope.$apply();
            }
        }
    }

}());
