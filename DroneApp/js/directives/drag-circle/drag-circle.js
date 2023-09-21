(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('dragCircle', ['$document', dragCircleDirective]);

    function dragCircleDirective($document) {
        return {
            scope: { angleMin: '=', range: '=', position: '=' },
            link: function (scope, element, attr) {
                var clientRect,centerX,centerY;
                var x,y,angleTrigo,radius,angle;
                element.on('mousedown', function (event) {
                    clientRect = element[0].getBoundingClientRect();
                    centerX = clientRect.left + (clientRect.width/2);
                    centerY = clientRect.top + (clientRect.height/2);
                    event.preventDefault();
                    x = (event.screenX - centerX)/(clientRect.width/2);
                    y = - (event.screenY - centerY)/(clientRect.height/2);
                    radius = Math.sqrt(x*x + y*y)*100;
                    if(radius>75 && radius < 90){
                        angleTrigo = Math.atan(y/x);
                        if(x<0){
                            angleTrigo = angleTrigo + Math.PI;
                        }
                        angle = scope.angleMin - (angleTrigo*180/Math.PI);
                        if(angle>0 && angle<scope.range){
                            scope.position = angle / (scope.range);  
                            scope.$apply();   
                            $document.on('mousemove', mousemove);
                            $document.on('mouseup', mouseup);
                        }
                    }
                });
                function mousemove(event) {
                    x = (event.screenX - centerX)*(53/100); 
                    y = - (event.screenY - centerY);
                    angleTrigo = Math.atan(y/x);
                    if(x<0){
                        angleTrigo = angleTrigo + Math.PI;
                    }
                    angle = scope.angleMin - (angleTrigo*180/Math.PI);
                    if(angle>0 && angle<scope.range){
                        scope.position = angle / (scope.range);        
                        scope.$apply();   
                    }      
                }

                function mouseup() {
                    $document.off('mousemove', mousemove);
                    $document.off('mouseup', mouseup);
                }
            }
        }
    }
} ());
