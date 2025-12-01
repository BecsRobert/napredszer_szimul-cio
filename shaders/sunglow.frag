#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

void main() {
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(TexCoord, center) * 2.0;
    
    float alpha = exp(-dist * 4.0); 
    
    if (dist > 1.0) discard;

    FragColor = vec4(1.0, 0.8, 0.3, alpha);
}