(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('winderControlCtrl', winderControlCtrl);

    function winderControlCtrl($scope) {
        var vm = this;
        var railLength=document.getElementsByClassName("rail-counter");
        $scope.updateLength  = function (value) {
            for (var i = 0; i < railLength.length; i++) {
                railLength[i].innerHTML=value.toFixed(1);
            }
        }
        $scope.stop  = function () {  
            $scope.mainControl=0;
            $scope.winderControl1=0;
            $scope.winderControl2=0;
            $scope.winderControl3=0;
            $scope.winderControl4=0;
        }
        $scope.reset  = function () {  
            $scope.winderControl1=0;
            $scope.winderControl2=0;
            $scope.winderControl3=0;
            $scope.winderControl4=0;
        }
        $scope.rail  = function () { 
            $scope.railMode=!$scope.railMode; 
            $scope.mainControl=0;
            $scope.winderControl1=0;
            $scope.winderControl2=0;
            $scope.winderControl3=0;
            $scope.winderControl4=0;
        }
    }

}());
