(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('rightSideCtrl', ['$timeout', '$scope', rightSideCtrl]);

    function rightSideCtrl($timeout, $scope) {
        var vm = this;

        $scope.depthSize = 145;
        $scope.depthThickness = 25;
        $scope.thrustSize = 280;
        $scope.thrustThickness = 30;

        $scope.focusRight = function (index, completeHandler) {
            if ($scope.rightData.focusRightIndex == index) return;
            $scope.rightData.positionClass[$scope.rightData.focusRightIndex] = $scope.rightData.positionClass[$scope.rightData.focusRightIndex] + " transition-right";
            var clickedItemClass = $scope.rightData.positionClass[index];
            $scope.rightData.positionClass[index] = $scope.rightData.positionClass[index] + " transition-right";
            $timeout(function () {
                $scope.rightData.positionClass[$scope.rightData.focusRightIndex] = $scope.rightData.focusOrigClass;
                $scope.rightData.focusOrigClass = clickedItemClass;
                $scope.rightData.positionClass[index] = "focus-right";
                $scope.rightData.focusRightIndex = index;
                if (focusRightHandler) {
                    focusRightHandler(false);
                }
                if (completeHandler) {
                    completeHandler(true);
                    focusRightHandler = completeHandler;
                }
            }, 300);
        };


        $scope.setThrustSize = function (focus) {
            if (focus) {
                $scope.thrustSize = 280;
                $scope.thrustThickness = 30;
            } else {
                $scope.thrustSize = 145;
                $scope.thrustThickness = 25;
            }
            $scope.$apply();
        };

        var focusRightHandler = $scope.setThrustSize;

        $scope.setDepthSize = function (focus) {
            if (focus) {
                $scope.depthSize = 280;
                $scope.depthThickness = 30;
            } else {
                $scope.depthSize = 145;
                $scope.depthThickness = 25;
            }
            $scope.$apply();
        };
    }

} ());
