(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('dragDirection', ['$document', dragDirectionDirective]);

    function dragDirectionDirective($document) {
        return {
            scope: {position: '=' },
            link: function (scope, element, attr) {
                var centerX,centerY;
                var x,y,angleTrigo;
                element.on('mousedown', function (event) {
                    event.preventDefault();
                    centerX = event.screenX ;
                    centerY = event.screenY;
                    x = event.screenX - centerX; 
                    y = - (event.screenY - centerY); 
                    $document.on('mousemove', mousemove);  
                    $document.on('mouseup', mouseup);
                });
                function mousemove(event) {
                    x = event.screenX - centerX; 
                    y = - (event.screenY - centerY);
                    angleTrigo = Math.atan(y/x)/Math.PI*180;
                    if(x<0){
                        angleTrigo = angleTrigo + 180;
                    } 
                    scope.position=270-angleTrigo; 
                    scope.$apply();                }

                function mouseup() {
                    $document.off('mousemove', mousemove);
                    $document.off('mouseup', mouseup);
                }
            }
        }
    }
} ());
