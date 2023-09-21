(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('leftSidePumpCtrl', ['$timeout', '$scope', leftSidePumpCtrl]);

    function leftSidePumpCtrl($timeout,$scope) {
        var vm = this,
            focusLeftHandler = null;
        
        $scope.pumpFocusLeft = function (index, completeHandler) {
            if ($scope.leftDataPump.focusLeftIndex == index) return;
            $scope.leftDataPump.positionClass[$scope.leftDataPump.focusLeftIndex] = $scope.leftDataPump.positionClass[$scope.leftDataPump.focusLeftIndex] + " transition-left";
            var clickedItemClass = $scope.leftDataPump.positionClass[index];
            $scope.leftDataPump.positionClass[index] = $scope.leftDataPump.positionClass[index] + " transition-left";
            $timeout(function () {
                $scope.leftDataPump.positionClass[$scope.leftDataPump.focusLeftIndex] = $scope.leftDataPump.focusOrigClass;
                $scope.leftDataPump.focusOrigClass = clickedItemClass;
                $scope.leftDataPump.positionClass[index] = "focus-left";
                $scope.leftDataPump.focusLeftIndex = index;
                if (focusLeftHandler) {
                    focusLeftHandler(false);
                }
                if (completeHandler) {
                    completeHandler(true);
                    focusLeftHandler = completeHandler;
                }
            }, 300);
        };

    }

}());
