(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('thrustCtrl', thrustCtrl);

    function thrustCtrl($scope) {
        var vm = this;
        var cadran=document.getElementsByClassName('cadrans-container thrust');
        for(var i=0; i<cadran.length; i++){
            cadran[i].addEventListener("wheel", scrollThrust, false);
        }
        function scrollThrust(event) {
            if($scope.focusIndex==0){
                if($scope.thrust>=0 && $scope.thrust<=1){
                    $scope.thrust=$scope.thrust+event.deltaY/10000;
                    if($scope.thrust>1){
                        $scope.thrust=1;
                    }
                    else if($scope.thrust<0){
                        $scope.thrust=0;
                    }
                    $scope.$apply();
                }  
            }
        }
    }

}());
