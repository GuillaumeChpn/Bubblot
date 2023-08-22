(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('pumpHorizonCtrl', pumpHorizonCtrl);

    function pumpHorizonCtrl($scope) {
        var vm = this;
        $scope.cursorPosition  = function () {
            if($scope.pitch > 45){
                $scope.pitch = 45;
            }
            if($scope.pitch < -45){
                $scope.pitch = -45;
            }
            return 23*Math.tan($scope.pitch*Math.PI/180);
        }
    }

}());
