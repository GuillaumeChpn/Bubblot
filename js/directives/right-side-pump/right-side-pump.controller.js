(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('rightSidePumpCtrl', ['$timeout', '$scope', rightSidePumpCtrl]);

    function rightSidePumpCtrl($timeout, $scope) {
        var vm = this;
        $scope.pumpDepthSize = 145;
        $scope.pumpDepthThickness = 25;
        $scope.pumpThrustSize = 280;
        $scope.pumpThrustThickness = 30;

        $scope.pumpFocusRight = function (index, completeHandler) {
            if ($scope.rightDataPump.focusRightIndex == index) return;
            $scope.rightDataPump.positionClass[$scope.rightDataPump.focusRightIndex] = $scope.rightDataPump.positionClass[$scope.rightDataPump.focusRightIndex] + " transition-right";
            var clickedItemClass = $scope.rightDataPump.positionClass[index];
            $scope.rightDataPump.positionClass[index] = $scope.rightDataPump.positionClass[index] + " transition-right";
            $timeout(function () {
                $scope.rightDataPump.positionClass[$scope.rightDataPump.focusRightIndex] = $scope.rightDataPump.focusOrigClass;
                $scope.rightDataPump.focusOrigClass = clickedItemClass;
                $scope.rightDataPump.positionClass[index] = "focus-right";
                $scope.rightDataPump.focusRightIndex = index;
                if (focusRightHandler) {
                    focusRightHandler(false);
                }
                if (completeHandler) {
                    completeHandler(true);
                    focusRightHandler = completeHandler;
                }
            }, 300);
        };


        $scope.setPumpThrustSize = function (focus) {
            if (focus) {
                $scope.pumpThrustSize = 280;
                $scope.pumpThrustThickness = 30;
            } else {
                $scope.pumpThrustSize = 145;
                $scope.pumpThrustThickness = 25;
            }
            $scope.$apply();
        };

        var focusRightHandler = $scope.setPumpThrustSize;

        $scope.setPumpDepthSize = function (focus) {
            if (focus) {
                $scope.pumpDepthSize = 280;
                $scope.pumpDepthThickness = 30;
            } else {
                $scope.pumpDepthSize = 145;
                $scope.pumpDepthThickness = 25;
            }
            $scope.$apply();
        };
    }

} ());
